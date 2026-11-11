"""Step D2 (CPU): from the base read, build one edit family the way Astra's
recipe does (docs/part1.html, 'How Astra found the direction', steps 3-6):
  direction : per layer, a difference of means (not held minus held) over a
              token set, on PRACTICE findings only; u_l for MLP layer l is the
              unit contrast at that block's output (residual l+1)
  layers    : growth of the contrast along u_l across block l, split into the
              MLP's part (u_l . contrast of the MLP output) and the rest
              (attention); the layers whose MLP writes most are edited
  columns   : score_j = (u_l^T w_j) * contrast of MLP activation j; top K
  control   : a random unit direction orthogonal to every u, on the same
              columns, its dose scaled so the weight change has the same norm

Usage: python d_build.py out/read OUT.json SET CAND K [LAYERS|auto[N]] [SEED]
SET in pre|ref|quote|item ; CAND in natural|matched|counterfactual
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import lib

src, dst, set_name, cand, K = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
layer_arg = sys.argv[6] if len(sys.argv) > 6 else "auto2"
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261110
s = lib.SETS.index(set_name)
meta = lib.read_jsonl(src / "meta.jsonl")
cf = lib.read_jsonl(src / "cfmeta.jsonl")
ok = np.array([m["ok"] for m in meta])
miss = np.array([not m["held"] for m in meta])
prac = np.array([m["split"] == "practice" for m in meta])
fsrc = np.array([m["source"] == "finding" for m in meta])
answer = np.array([hash((m["run"], m["key"])) for m in meta])
row_of = {(m["run"], m["key"], m["index"]): m["row"] for m in meta}
cf_rows = np.array([row_of[(c["run"], c["key"], c["index"])] for c in cf])
cf_prac = np.array([c["split"] == "practice" for c in cf])
use = ok & fsrc & prac


def contrast(name):
    """Practice contrast of one stored quantity ('H', 'M' or 'A'): [layers, width]."""
    X = np.load(src / f"{name}.npy", mmap_mode="r")
    if cand == "counterfactual":
        C = np.load(src / f"c{name}.npy", mmap_mode="r")
        rows = cf_rows[cf_prac]
        a = np.stack([np.asarray(X[r, s], np.float32) for r in rows])
        b = np.asarray(C[cf_prac, s - 1], np.float32)
        return (a - b).mean(0), len(rows)
    if cand == "natural":
        a = np.stack([np.asarray(X[r, s], np.float32) for r in np.where(use & miss)[0]]).mean(0)
        b = np.stack([np.asarray(X[r, s], np.float32) for r in np.where(use & ~miss)[0]]).mean(0)
        return a - b, int((use & miss).sum())
    groups = defaultdict(list)
    for r in np.where(use)[0]:
        groups[answer[r]].append(r)
    diffs = []
    for g in groups.values():
        p = [r for r in g if miss[r]]
        n = [r for r in g if not miss[r]]
        if p and n:
            diffs.append(
                np.stack([np.asarray(X[r, s], np.float32) for r in p]).mean(0)
                - np.stack([np.asarray(X[r, s], np.float32) for r in n]).mean(0)
            )
    return np.mean(diffs, 0), len(diffs)


dH, n_pairs = contrast("H")  # [33, D]
dM, _ = contrast("M")  # [32, D]
dA, _ = contrast("A")  # [32, I]
L = lib.N_LAYERS
table = []
U = {}
for l in range(L):
    u = dH[l + 1] / np.linalg.norm(dH[l + 1])
    U[l] = u
    growth = float(u @ (dH[l + 1] - dH[l]))
    mlp = float(u @ dM[l])
    table.append({"layer": l, "contrast_norm_out": float(np.linalg.norm(dH[l + 1])),
                  "growth_along_u": growth, "mlp_part": mlp, "attention_part": growth - mlp})
if layer_arg.startswith("auto"):
    n = int(layer_arg[4:] or 2)
    layers = sorted(sorted(range(L), key=lambda l: -table[l]["mlp_part"])[:n])
else:
    layers = [int(x) for x in layer_arg.split(",")]
index = json.load(open(lib.MODEL / "model.safetensors.index.json"))["weight_map"]
rng = torch.Generator().manual_seed(seed)
basis = torch.linalg.qr(torch.tensor(np.stack([U[l] for l in layers], 1), dtype=torch.float32)).Q
family = {"name": Path(dst).stem, "set": set_name, "candidate": cand, "K": K,
          "n_contrast_units": n_pairs, "seed": seed, "doses": [1, 2, 3, 4, 5, 6],
          "layer_table": table, "layers_chosen": layers,
          "target": {"layers": {}}, "control": {"layers": {}}, "per_layer": {}}
for l in layers:
    name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
    with safe_open(str(lib.MODEL / index[name]), "pt") as f:
        W = f.get_tensor(name).float()  # [D, I]
    u = torch.tensor(U[l], dtype=torch.float32)
    u = u / u.norm()
    write = u @ W  # [I]
    score = write.numpy() * dA[l]
    order = np.argsort(-score)
    cols = sorted(int(j) for j in order[:K])
    pos_mass = float(score[score > 0].sum())
    v = torch.randn(lib.HIDDEN, generator=rng)
    v = v - basis @ (basis.T @ v)
    v = v / v.norm()
    Wc = W[:, cols]
    scale = float((u @ Wc).norm() / (v @ Wc).norm())
    top_write = set(int(j) for j in np.argsort(-np.abs(write.numpy()))[:K])
    family["target"]["layers"][str(l)] = {"u": u.tolist(), "columns": cols}
    family["control"]["layers"][str(l)] = {"u": v.tolist(), "columns": cols, "alpha_scale": scale}
    family["per_layer"][str(l)] = {
        "share_of_positive_score_mass": float(score[order[:K]].sum() / pos_mass),
        "columns_also_in_top_K_by_write_strength": len(top_write & set(cols)),
        "rel_frobenius_change_per_unit_dose": float((u @ Wc).norm() / W.norm()),
        "control_alpha_scale": scale,
        "control_cos_with_u": float(v @ u),
        "u_dot_previous_layer_u": None,
    }
ls = [str(l) for l in layers]
for a, b in zip(ls, ls[1:]):
    family["per_layer"][b]["u_dot_previous_layer_u"] = float(
        np.dot(family["target"]["layers"][a]["u"], family["target"]["layers"][b]["u"]))
json.dump(family, open(dst, "w"))
print("contrast units", n_pairs, "layers chosen", layers)
print("layer  |contrast|  growth   mlp    attention")
for t in table:
    print(f"{t['layer']:5d} {t['contrast_norm_out']:9.3f} {t['growth_along_u']:8.3f} {t['mlp_part']:7.3f} {t['attention_part']:8.3f}")
print(json.dumps(family["per_layer"], indent=1))
