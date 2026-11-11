"""Grader check; exits 1 on ANY disagreement or failed case (the first version's check printed and exited 0).

Three parts, all on files released by the authors (CPU):
 A. equivalence: grader.equiv on the authors' extracted first answer (pred_first) against their math-verify flag
    (correct[0]), every MATH-500 problem of the unedited and p50c runs that has a pred_first.
 B. extraction: grader.extract on the released first-sample TEXTS (20 per shard) must give the authors' pred_first
    (compared after grader.norm; both None counts as agreement).
 C. unit cases: the SymPy path, the base-subscript rule, the supplementary extraction.
It also prints how many answers part A skipped (no pred_first), since those are not checked by A.
Usage: python -I check_grader.py LOGS_EXP8_DIR MATH500_JSONL
(LOGS_EXP8_DIR is exp8_math/ of the authors' logs repository; a Python with sympy 1.14.0.)
"""
import glob
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader as G  # noqa: E402

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
LOGS = sys.argv[1]
gold = {json.loads(l)["unique_id"]: json.loads(l)["answer"] for l in open(sys.argv[2])}
bad = []
agree = skipped = ex_agree = ex_n = 0
for run in ("unedited", "p50c"):
    for d in sorted(glob.glob(f"{LOGS}/eval/{run}/parts/math500_*")):
        per = {r["id"]: r for r in json.load(open(f"{d}/math500_per_problem.json"))}
        for r in per.values():
            p = r["pred_first"]
            if p is None:
                skipped += 1
                continue
            mine, theirs = G.equiv(p, gold[r["id"]]), bool(r["correct"][0])
            if mine == theirs:
                agree += 1
            else:
                bad.append(("A equivalence", run, r["id"], p[:60], gold[r["id"]], "theirs", theirs, "mine", mine))
        for line in gzip.open(f"{d}/math500_first_samples.jsonl.gz", "rt"):
            s = json.loads(line)
            ex_n += 1
            mine, theirs = G.extract(s["text"]), per[s["id"]]["pred_first"]
            if (mine is None) == (theirs is None) and (mine is None or G.norm(mine) == G.norm(theirs)):
                ex_agree += 1
            else:
                bad.append(("B extraction", run, s["id"], "theirs", theirs, "mine", mine))
CASES = [("10/11", "\\frac{10}{11}", True), ("639/40", "\\frac{639}{40}", True), ("5/13", "\\frac{5}{13}", True),
         ("(x+2)(x+1)", "(x+1)(x+2)", True), ("52", "52_8", True), ("4210_{5}", "4210_5", True),
         ("52_9", "52_8", False), ("x_1", "x_2", False), ("x_1", "x", False), ("3", "4", False),
         ("0.5", "\\frac{1}{2}", True), ("2\\sqrt{2}", "\\sqrt{8}", True), ("20\\text{ miles}", "20", True)]
for p, g, want in CASES:
    if G.equiv(p, g) != want:
        bad.append(("C unit", p, g, "want", want))
SUPP = [("thinking</think>\nSo we get it.\n**Final Answer:** 42", "42"), ("</think>Answer: $\\frac{1}{2}$.", "\\frac{1}{2}"),
        ("</think>\\boxed{7}\nAnswer: 8", "7"), ("</think>no answer here", None)]
for t, want in SUPP:
    if G.extract_supp(t) != want:
        bad.append(("C supp", t, "got", G.extract_supp(t), "want", want))
print(f"A equivalence: agree {agree}, disagree {sum(b[0] == 'A equivalence' for b in bad)}, skipped (no pred_first) {skipped}")
print(f"B extraction: agree {ex_agree} of {ex_n} released texts, disagree {sum(b[0] == 'B extraction' for b in bad)}")
print(f"C unit cases: {len(CASES) + len(SUPP)} cases, failed {sum(b[0].startswith('C') for b in bad)}")
for b in bad:
    print("DISAGREE", b)
sys.exit(1 if bad else 0)
