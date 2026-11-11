"""Compare grader.equiv with the authors' math-verify flags on their released first answers (pred_first, correct[0]).
Usage: python -I check_grader.py GRADER_DIR LOGS_EXP8_DIR MATH500_JSONL
GRADER_DIR holds grader.py; LOGS_EXP8_DIR is exp8_math/ of the authors' logs repository."""
import glob, json, sys
sys.path.insert(0, sys.argv[1])
import grader as G
gold = {json.loads(l)["unique_id"]: json.loads(l)["answer"] for l in open(sys.argv[3])}
agree = dis = 0; rows = []
for run in ("unedited", "p50c"):
    for f in sorted(glob.glob(f"{sys.argv[2]}/eval/{run}/parts/math500_*/math500_per_problem.json")):
        for r in json.load(open(f)):
            p = r["pred_first"]
            if p is None:
                continue
            mine = G.equiv(p, gold[r["id"]]); theirs = bool(r["correct"][0])
            if mine == theirs: agree += 1
            else: dis += 1; rows.append((run, r["id"], p[:60], gold[r["id"]], theirs, mine))
print("agree", agree, "disagree", dis)
for x in rows: print(x)
