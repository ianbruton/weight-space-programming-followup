"""Write an edited copy of the 4B as a new model folder (CPU, low memory).
The base files are opened read-only and never changed: the new folder holds
symlinks to every base file except the weight shard(s) that hold an edited
tensor; each of those is copied byte for byte and the edited tensors' bytes
are then overwritten in the copy (same safetensors header and layout).
Edit: W <- W0 - alpha * u (u^T W0) on the listed columns of
model.language_model.layers.<l>.mlp.down_proj.weight (float32 arithmetic from
the bf16 base, cast back to bf16). Verifies the copy against the base byte
range by byte range and prints the folder's manifest SHA-256 (the sha256 of
the list of '<file sha256>  <name>' lines, so a served model can be pinned).

Usage: python e_write.py EDIT.json OUT_DIR
EDIT.json: {"name":..., "alpha": a, "layers": {"<l>": {"u": [...], "columns": [...], ["alpha_scale": s]}}}
"""

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lib

CHUNK = 1 << 24
edit = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
index = json.load(open(lib.MODEL / "model.safetensors.index.json"))["weight_map"]
names = {
    f"model.language_model.layers.{int(l)}.mlp.down_proj.weight": spec
    for l, spec in edit["layers"].items()
}
shards = sorted({index[n] for n in names})
report = {"name": edit.get("name"), "alpha": edit["alpha"], "tensors": {}}
for item in sorted(lib.MODEL.iterdir()):
    if item.name in shards or item.name.startswith("."):
        continue
    os.symlink(item.resolve(), out / item.name)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


for shard in shards:
    src, dst = lib.MODEL / shard, out / shard
    with open(src, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    data0 = 8 + hlen
    with open(src, "rb") as a, open(dst, "wb") as b:
        for chunk in iter(lambda: a.read(CHUNK), b""):
            b.write(chunk)
    ranges = []
    with open(src, "rb") as a, open(dst, "r+b") as b:
        for n, spec in names.items():
            if index[n] != shard:
                continue
            info = header[n]
            assert info["dtype"] == "BF16" and info["shape"] == [lib.HIDDEN, lib.INTER]
            lo, hi = data0 + info["data_offsets"][0], data0 + info["data_offsets"][1]
            a.seek(lo)
            raw = np.frombuffer(a.read(hi - lo), dtype=np.int16).reshape(lib.HIDDEN, lib.INTER)
            w0 = torch.from_numpy(raw.copy()).view(torch.bfloat16)
            u = torch.tensor(spec["u"], dtype=torch.float32)
            assert abs(float(u.norm()) - 1) < 1e-4
            cols = torch.tensor(spec["columns"])
            w32 = w0.float()
            sub = w32[:, cols]
            sub = sub - edit["alpha"] * spec.get("alpha_scale", 1.0) * torch.outer(u, u @ sub)
            w32[:, cols] = sub
            w1 = w32.to(torch.bfloat16)
            changed = w1.view(torch.int16) != w0.view(torch.int16)
            other = torch.ones(lib.INTER, dtype=torch.bool)
            other[cols] = False
            assert not changed[:, other].any()
            new = w1.view(torch.int16).numpy().tobytes()
            assert len(new) == hi - lo
            b.seek(lo)
            b.write(new)
            ranges.append((lo, hi, n))
            report["tensors"][n] = {
                "shard": shard,
                "columns": len(spec["columns"]),
                "dose_applied": edit["alpha"] * spec.get("alpha_scale", 1.0),
                "entries_changed": int(changed.sum()),
                "entries_addressed": int(len(cols) * lib.HIDDEN),
                "rel_frobenius_change": float((w1.float() - w0.float()).norm() / w0.float().norm()),
                "base_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "edited_sha256": hashlib.sha256(new).hexdigest(),
            }
    # verify from disk: outside the edited tensors' byte ranges the copy equals the base
    ranges.sort()
    differing = 0
    with open(src, "rb") as a, open(dst, "rb") as b:
        pos = 0
        while True:
            x, y = a.read(CHUNK), b.read(CHUNK)
            if not x and not y:
                break
            assert len(x) == len(y)
            if x != y:
                xa, ya = np.frombuffer(x, np.uint8), np.frombuffer(y, np.uint8)
                idx = np.nonzero(xa != ya)[0] + pos
                inside = np.zeros(len(idx), bool)
                for lo, hi, _ in ranges:
                    inside |= (idx >= lo) & (idx < hi)
                assert inside.all(), "a byte outside the edited tensors differs"
                differing += len(idx)
            pos += len(x)
    assert differing > 0
    report.setdefault("verified_shards", []).append({"shard": shard, "bytes_differing": int(differing)})
# manifest: sha256 of every file as the harness sees it (symlinks followed);
# base files' hashes are cached, they never change
cache_file = out.parent / "base-sha256.json"
cache = json.load(open(cache_file)) if cache_file.exists() else {}
lines = []
for item in sorted(out.iterdir(), key=lambda p: p.name):
    if item.name.startswith(".") or not item.is_file():
        continue
    if item.is_symlink():
        real = str(item.resolve())
        if real not in cache:
            cache[real] = file_sha(real)
        digest = cache[real]
    else:
        digest = file_sha(item)
    lines.append(f"{digest}  {item.name}\n")
cache_file.write_text(json.dumps(cache, indent=1))
report["manifest_sha256"] = hashlib.sha256("".join(lines).encode()).hexdigest()
report["manifest"] = [x.strip() for x in lines]
# the recipe travels beside the folder, not inside it (the manifest covers the folder)
side = out.parent / (out.name + ".json")
json.dump({"edit": edit, "report": report}, open(side, "w"))
print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=1))
