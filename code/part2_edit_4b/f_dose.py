"""Step F (GPU, one job): teacher-forced dose screen, before any
generation. For the base and for every edit of a family (target direction and
its norm-matched random control, doses 1-6), every recorded final answer is
read once (the prompt as vLLM saw it + the answer the base 4B wrote) and kept:
  per answer   : mean log-probability of the answer's tokens (general damage)
  per citation : log-probability of its ref and of its quote
  per path-wrong miss: at the first token where the 4B's wrong ref and a ref
                 that holds part ways, the log-probability of each next token
The edits are applied in memory with the arithmetic of the weight files.

Usage: python f_dose.py data/final.jsonl FAMILY.json OUT.json [practice|all]
FAMILY.json: {"target": {"layers": {l: {"u","columns"}}},
              "control": {"layers": {l: {"u","columns","alpha_scale"}}}, "doses": [1..6]}
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import lib

recs = lib.read_jsonl(sys.argv[1])
family = json.load(open(sys.argv[2]))
out_path = sys.argv[3]
which = sys.argv[4] if len(sys.argv) > 4 else "all"
if which in ("practice", "lite"):
    recs = [r for r in recs if lib.split_of(r["key"]) == "practice"]
if which == "lite":
    # every practice answer with a miss, and every third of the clean ones
    withmiss = [r for r in recs if any(not it["held"] for it in r["items"])]
    clean = [r for r in recs if all(it["held"] for it in r["items"])]
    recs = withmiss + clean[::3]
tok, model = lib.load()

prepared = []
for r in recs:
    p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    enc = tok(r["answer"], add_special_tokens=False, return_offsets_mapping=True)
    a_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    items = []
    for it in r["items"]:
        if not isinstance(it["item"], dict):
            continue
        pos = lib.item_positions(offsets, it)
        rec = {"index": it["index"], "held": it["held"], "why": it["why"],
               "source": it["item"].get("source"), "ref_closed": pos["ref_closed"],
               "quote": pos["quote"]}
        if it.get("paths_holding"):
            good = it["paths_holding"][0]
            r0, r1 = it["ref"][0] + 1, it["ref"][1] - 1
            text = r["answer"][:r0] + good + r["answer"][r1 : it["item_span"][1]]
            b_ids = tok(text, add_special_tokens=False)["input_ids"]
            d = 0
            while d < min(len(a_ids), len(b_ids)) and a_ids[d] == b_ids[d]:
                d += 1
            rec["fork"] = {"at": d, "wrong": a_ids[d], "right": b_ids[d]}
        items.append(rec)
    prepared.append({"run": r["run"], "key": r["key"], "split": lib.split_of(r["key"]),
                     "p_ids": p_ids, "a_ids": a_ids, "items": items})


@torch.no_grad()
def read_all():
    rows = []
    for p in prepared:
        ids, P = p["p_ids"] + p["a_ids"], len(p["p_ids"])
        x = torch.tensor([ids], device=model.device)
        last = model.model(input_ids=x, use_cache=False).last_hidden_state[0, P - 1 : -1]
        target = x[0, P:]
        lps = []
        forks = {it["fork"]["at"] for it in p["items"] if "fork" in it}
        fork_rows = {}
        for i in range(0, last.shape[0], 256):
            lp = torch.log_softmax(model.lm_head(last[i : i + 256]).float(), -1)
            lps.append(lp.gather(1, target[i : i + 256, None])[:, 0])
            for d in forks:
                if i <= d < i + 256:
                    fork_rows[d] = lp[d - i].clone()
        lp = torch.cat(lps).cpu()
        items = []
        for it in p["items"]:
            rec = {"index": it["index"], "ref_lp": float(lp[it["ref_closed"]].sum()),
                   "quote_lp": float(lp[it["quote"]].sum())}
            if "fork" in it:
                row = fork_rows[it["fork"]["at"]]
                rec["fork_wrong_lp"] = float(row[it["fork"]["wrong"]])
                rec["fork_right_lp"] = float(row[it["fork"]["right"]])
                rec["fork_top_is_right"] = int(row.argmax()) == it["fork"]["right"]
                rec["fork_top_is_wrong"] = int(row.argmax()) == it["fork"]["wrong"]
            items.append(rec)
        rows.append({"run": p["run"], "key": p["key"], "mean_lp": float(lp.mean()),
                     "tokens": len(lp), "items": items})
    return rows


# FAMILY.json may be one family, or a plan: {"plan": [{"family": path, "target": [doses], "control": [doses]}]}
fams = {}
conds = [("base", None, None, 0)]
if "plan" in family:
    for row in family["plan"]:
        fam = json.load(open(row["family"]))
        fams[fam["name"]] = fam
        conds += [(fam["name"], "target", fam["name"], a) for a in row["target"]]
        conds += [(fam["name"], "control", fam["name"], a) for a in row["control"]]
else:
    fams[family["name"]] = family
    for a in family.get("doses", [1, 2, 3, 4, 5, 6]):
        conds.append((family["name"], "target", family["name"], a))
        conds.append((family["name"], "control", family["name"], a))
result = {"which": which, "conditions": []}
static = [{"run": p["run"], "key": p["key"], "split": p["split"],
           "items": [{k: v for k, v in it.items() if k in ("index", "held", "why", "source")}
                     | ({"fork": it["fork"]} if "fork" in it else {}) for it in p["items"]]}
          for p in prepared]
result["static"] = static
for fname, arm, fam, alpha in conds:
    t0 = time.time()
    saved = None
    if arm:
        saved = lib.apply_edit(model, {"alpha": alpha, "layers": fams[fam][arm]["layers"]})
    rows = read_all()
    if saved is not None:
        lib.restore(model, saved)
    result["conditions"].append({"family": fname, "arm": arm or "base", "alpha": alpha, "rows": rows})
    print(fname, arm, alpha, round(time.time() - t0, 1), "s", flush=True)
    json.dump(result, open(out_path, "w"))
print("done")
