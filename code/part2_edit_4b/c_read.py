"""Step C (GPU, one job): teacher-forced read of the base 4B over every
recorded final answer (prompt exactly as vLLM saw it + the answer the 4B
wrote). Keeps, per citation and per layer, the residual stream, the MLP output
and the MLP inner activation, averaged over four token sets (pre, ref, quote,
item), and the log-probability of the ref. For each path-wrong miss a second
branch is read with the ref replaced by a path that holds (same prefix, same
quote).

Usage: python c_read.py data/final.jsonl out/read
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lib

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
recs = lib.read_jsonl(src)
tok, model = lib.load()
taps = lib.Taps(model)
n_items = sum(len(r["items"]) for r in recs)
n_cf = sum(1 for r in recs for it in r["items"] if it.get("paths_holding"))
L, D, I = lib.N_LAYERS, lib.HIDDEN, lib.INTER
H = np.zeros((n_items, 4, L + 1, D), np.float32)
M = np.zeros((n_items, 4, L, D), np.float32)
A = np.zeros((n_items, 4, L, I), np.float16)
cH = np.zeros((n_cf, 3, L + 1, D), np.float32)
cM = np.zeros((n_cf, 3, L, D), np.float32)
cA = np.zeros((n_cf, 3, L, I), np.float16)


def gather(store, n, rows):
    """[n layers, width]: mean over the rows (tap rows are answer index + 0,
    because tap row r is the state at answer token r - 1 + 1... see below)."""
    idx = torch.tensor(rows, device=model.device)
    return torch.stack(
        [store[i].index_select(0, idx).float().mean(0) for i in n]
    ).cpu().numpy()


meta, answers, cfmeta = [], [], []
row = crow = 0
t0 = time.time()
for ri, r in enumerate(recs):
    p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    enc = tok(r["answer"], add_special_tokens=False, return_offsets_mapping=True)
    a_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    assert len(p_ids) == r["prompt_tokens_vllm"]
    P = len(p_ids)
    lp, ent = lib.forward(model, p_ids + a_ids, P, taps)
    # tap row k is the state at sequence position P - 1 + k, i.e. the state
    # AT answer token k - 1; the state at answer token t is row t + 1.
    answers.append(
        {
            "run": r["run"],
            "key": r["key"],
            "split": lib.split_of(r["key"]),
            "tokens": len(a_ids),
            "mean_logprob": float(lp.mean()),
            "sum_logprob": float(lp.sum()),
        }
    )
    for it in r["items"]:
        pos = lib.item_positions(offsets, it) if isinstance(it["item"], dict) else None
        m = {
            "row": row,
            "run": r["run"],
            "key": r["key"],
            "kind": r["kind"],
            "split": lib.split_of(r["key"]),
            "index": it["index"],
            "held": it["held"],
            "why": it["why"],
            "source": it["item"].get("source") if isinstance(it["item"], dict) else None,
            "ref": it["item"].get("ref") if isinstance(it["item"], dict) else None,
            "quote": it["item"].get("quote") if isinstance(it["item"], dict) else None,
            "ok": pos is not None,
        }
        if pos is not None:
            for s, name in enumerate(lib.SETS):
                rows = [t + 1 for t in pos[name]]
                H[row, s] = gather(taps.h, range(L + 1), rows)
                M[row, s] = gather(taps.m, range(L), rows)
                A[row, s] = gather(taps.a, range(L), rows).astype(np.float16)
            m["n_tok"] = {k: len(v) for k, v in pos.items()}
            m["ref_logprob"] = float(lp[pos["ref_closed"]].sum())
            m["first_logprob"] = float(lp[pos["pre"][0] + 1])
            m["pre_entropy"] = float(ent[pos["pre"][0] + 1])
            m["quote_logprob"] = float(lp[pos["quote"]].sum())
        meta.append(m)
        row += 1
    # the corrected-ref branches of this answer's path-wrong misses
    for it in r["items"]:
        if not it.get("paths_holding"):
            continue
        good = it["paths_holding"][0]
        r0, r1 = it["ref"][0] + 1, it["ref"][1] - 1
        text = r["answer"][:r0] + good + r["answer"][r1 : it["item_span"][1]]
        enc2 = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        b_ids, off2 = enc2["input_ids"], enc2["offset_mapping"]
        d = 0
        while d < min(len(a_ids), len(b_ids)) and a_ids[d] == b_ids[d]:
            d += 1
        shift = len(good) - (r1 - r0)
        it2 = {
            "ref": [it["ref"][0], it["ref"][1] + shift],
            "quote": [it["quote"][0] + shift, it["quote"][1] + shift],
            "item_span": [it["item_span"][0], it["item_span"][1] + shift],
        }
        pos2 = lib.item_positions(off2, it2)
        pos1 = lib.item_positions(offsets, it)
        lp2, _ = lib.forward(model, p_ids + b_ids, P, taps)
        for s, name in enumerate(lib.SETS[1:]):
            rows = [t + 1 for t in pos2[name]]
            cH[crow, s] = gather(taps.h, range(L + 1), rows)
            cM[crow, s] = gather(taps.m, range(L), rows)
            cA[crow, s] = gather(taps.a, range(L), rows).astype(np.float16)
        e1, e2 = pos1["ref_closed"][-1], pos2["ref_closed"][-1]
        cfmeta.append(
            {
                "crow": crow,
                "run": r["run"],
                "key": r["key"],
                "split": lib.split_of(r["key"]),
                "index": it["index"],
                "wrong_ref": it["item"]["ref"],
                "right_ref": good,
                "quote": it["item"]["quote"],
                "diverge_at": d,
                "wrong_tokens": tok.convert_ids_to_tokens(a_ids[d : e1 + 1]),
                "right_tokens": tok.convert_ids_to_tokens(b_ids[d : e2 + 1]),
                "lp_wrong": float(lp[d : e1 + 1].sum()),
                "lp_right": float(lp2[d : e2 + 1].sum()),
                "lp_wrong_first": float(lp[d]),
                "lp_right_first": float(lp2[d]),
                "lp_quote_after_wrong": float(lp[pos1["quote"]].sum()),
                "lp_quote_after_right": float(lp2[pos2["quote"]].sum()),
            }
        )
        crow += 1
    if ri % 10 == 0:
        print(ri, len(recs), P, len(a_ids), round(time.time() - t0, 1), flush=True)
assert row == n_items and crow == n_cf
np.save(out / "H.npy", H)
np.save(out / "M.npy", M)
np.save(out / "A.npy", A)
np.save(out / "cH.npy", cH)
np.save(out / "cM.npy", cM)
np.save(out / "cA.npy", cA)
for name, rows in (("meta", meta), ("answers", answers), ("cfmeta", cfmeta)):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
print("done", row, crow, round(time.time() - t0, 1), "s; peak GPU MiB",
      torch.cuda.max_memory_allocated() // 2**20)
