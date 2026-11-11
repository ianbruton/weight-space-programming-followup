"""Score gen_job.py output folders on MATH-500 with grader.py, and pair folders that share problems.
Usage: MATH500=path/to/test.jsonl python3 score.py DIR [DIR ...] [--json OUT]
Per folder: problems and generations scored, accuracy (the authors' extraction) with its standard error over problems,
the strict accuracy (thinking texts must close </think>), the share cut at max_tokens, the mean length, the share of cut
answers whose end loops, and the throughput the job recorded. Pairs (same setting, labels base_* / edited_* or
base_* / final_*): the paired difference over the problems both have, with its standard error.
"""
import glob
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader as G  # noqa: E402

MATH500 = os.environ.get("MATH500", "math500/test.jsonl")  # HuggingFaceH4/MATH-500 test.jsonl
GOLD = {json.loads(line)["unique_id"]: json.loads(line) for line in open(MATH500)}


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd / n ** 0.5


def score_dir(d):
    meta = json.load(open(Path(d) / "meta.json"))
    thinking = meta["job"]["thinking"]
    per, rows = {}, []
    for f in sorted(glob.glob(str(Path(d) / "chunk_*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            g = G.grade(r["text"], GOLD[r["id"]]["answer"], thinking)
            cut = r["finish"] == "length"
            loop = cut and (G.verbatim_loop_end(r["text"]) or G.repetitive_end(r["text"]))
            rows.append({"id": r["id"], "k": r["k"], "correct": g["correct"], "strict": g["correct_strict"],
                         "cut": cut, "loop": loop, "n_tokens": r["n_tokens"], "pred": g["pred"],
                         "closed_think": "</think>" in r["text"]})
            per.setdefault(r["id"], []).append(g["correct"])
    n = len(rows)
    if not n:
        return {"dir": str(d), "generations": 0}, {}
    acc, se = mean_se([sum(v) / len(v) for v in per.values()])
    cut = sum(r["cut"] for r in rows)
    ch = meta.get("chunks", {}).values()
    secs = sum(c["seconds"] for c in ch)
    toks = sum(c["tokens"] for c in ch)
    s = {"dir": os.path.basename(str(d)), "setting": meta["job"]["setting"], "model": meta["job"]["model"],
         "max_tokens": meta["job"]["sampling"]["max_tokens"], "problems": len(per), "generations": n,
         "samples_per_problem": round(n / len(per), 2),
         "accuracy": round(100 * acc, 2), "se": round(100 * se, 2),
         "accuracy_strict": round(100 * sum(r["strict"] for r in rows) / n, 2),
         "cut_share": round(100 * cut / n, 2), "mean_tokens": round(sum(r["n_tokens"] for r in rows) / n),
         "median_tokens": sorted(r["n_tokens"] for r in rows)[n // 2],
         "cut_with_loop_end": round(100 * sum(r["loop"] for r in rows) / cut, 1) if cut else None,
         "no_answer_finished": round(100 * sum(1 for r in rows if not r["cut"] and r["pred"] is None) / n, 2),
         "correct_among_cut": sum(1 for r in rows if r["cut"] and r["correct"]),
         "gen_seconds": round(secs), "tokens": toks, "tok_per_s": round(toks / secs) if secs else None}
    return s, per


def main():
    args = sys.argv[1:]
    js = None
    if "--json" in args:
        i = args.index("--json")
        js = args[i + 1]
        args = args[:i] + args[i + 2:]
    res, pers = [], {}
    for d in args:
        s, per = score_dir(d)
        res.append(s)
        pers[os.path.basename(d.rstrip("/"))] = per
    pairs = []
    for a in pers:
        if not a.startswith("base_"):
            continue
        for other in ("edited_", "final_"):
            b = other + a[len("base_"):]
            if b in pers:
                common = sorted(set(pers[a]) & set(pers[b]))
                diffs = [sum(pers[b][i]) / len(pers[b][i]) - sum(pers[a][i]) / len(pers[a][i]) for i in common]
                m, se = mean_se(diffs)
                if m is not None:
                    pairs.append({"base": a, "other": b, "problems": len(common), "diff_points": round(100 * m, 2),
                                  "se_points": round(100 * se, 2)})
    out = {"folders": res, "pairs": pairs}
    print(json.dumps(out, indent=1))
    if js:
        json.dump(out, open(js, "w"), indent=1)


if __name__ == "__main__":
    main()
