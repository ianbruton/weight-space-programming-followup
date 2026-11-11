"""Pinned scorer: score gen_job.py output folders on MATH-500 with grader.py.

Usage: MATH500=path/to/test.jsonl python -I score.py --out OUT.json [--rows DIR] [--ids FILE] LABEL=DIR [LABEL=DIR ...]
           [--pair A,B ...]
Run it with a Python that has sympy 1.14.0. It stops with a traceback (exit 1) if SymPy is missing or is not the
pinned version, if a folder has no generations, or if a pair shares no problems.

Per folder: problems, generations, accuracy by the authors' extraction (main), by the strict extraction, by the
supplementary extraction (a plain final 'Answer: X' line also read) and by the first version's base-subscript rule; the share cut
at max_tokens; mean tokens; finished answers with no box. The standard error is over problems (samples of one problem
are averaged first).
Per pair A,B (B minus A over the problems both have): difference in points, its standard error, an approximate 95%
interval (difference +- 1.96 se, unadjusted), the number of problems where only A / only B is better, and for
one-sample pairs the exact two-sided McNemar p.
--ids FILE restricts every folder to the listed problem ids. --rows DIR writes one JSON line per generation.
"""
import glob
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grader as G  # noqa: E402  (raises if SymPy is missing or not the pinned version)

MATH500 = os.environ.get("MATH500", "math500/test.jsonl")  # HuggingFaceH4/MATH-500 test.jsonl
GOLD = {json.loads(line)["unique_id"]: json.loads(line) for line in open(MATH500)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def self_test():
    """the scorer refuses to run if the SymPy path or the base rule is not live."""
    assert G.equiv("10/11", "\\frac{10}{11}"), "sympy numeric path dead"
    assert G.equiv("(x+2)(x+1)", "(x+1)(x+2)"), "sympy symbolic path dead"
    assert G.equiv("52", "52_8") and not G.equiv("52_9", "52_8") and not G.equiv("x_1", "x_2"), "base rule"
    assert not G.equiv("3", "4") and not G.equiv(None, "4")


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd / n ** 0.5


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def score_dir(label, d, ids, rows_dir):
    meta = json.load(open(Path(d) / "meta.json"))
    thinking = meta["job"]["thinking"]
    per, rows = {}, []
    files = sorted(glob.glob(str(Path(d) / "chunk_*.jsonl.gz")))
    for f in files:
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if ids is not None and r["id"] not in ids:
                continue
            g = G.grade(r["text"], GOLD[r["id"]]["answer"], thinking)
            cut = r["finish"] == "length"
            loop = cut and (G.verbatim_loop_end(r["text"]) or G.repetitive_end(r["text"]))
            rows.append({"id": r["id"], "k": r["k"], "correct": g["correct"], "strict": g["correct_strict"],
                         "supp": g["correct_supp"], "stage1": g["correct_stage1"], "cut": cut, "loop": loop,
                         "n_tokens": r["n_tokens"], "pred": g["pred"], "pred_supp": g["pred_supp"],
                         "gold": GOLD[r["id"]]["answer"], "closed_think": "</think>" in r["text"]})
            per.setdefault(r["id"], []).append(g["correct"])
    n = len(rows)
    if not n:
        raise SystemExit(f"{label}: no generations in {d}")
    if rows_dir:
        Path(rows_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(rows_dir) / f"{label}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    acc, se = mean_se([sum(v) / len(v) for v in per.values()])
    cut = sum(r["cut"] for r in rows)
    pct = lambda key: round(100 * sum(r[key] for r in rows) / n, 2)  # noqa: E731
    s = {"label": label, "dir": str(d), "setting": meta["job"]["setting"], "model": meta["job"]["model"],
         "max_tokens": meta["job"]["sampling"]["max_tokens"], "chunk_files": len(files), "problems": len(per),
         "generations": n, "samples_per_problem": round(n / len(per), 2),
         "accuracy": round(100 * acc, 2), "se": round(100 * se, 2), "correct_count": sum(r["correct"] for r in rows),
         "accuracy_strict": pct("strict"), "accuracy_supp": pct("supp"), "accuracy_stage1_base_rule": pct("stage1"),
         "cut_share": round(100 * cut / n, 2), "mean_tokens": round(sum(r["n_tokens"] for r in rows) / n),
         "cut_with_loop_end": round(100 * sum(r["loop"] for r in rows) / cut, 1) if cut else None,
         "no_box_finished": sum(1 for r in rows if not r["cut"] and r["pred"] is None)}
    return s, per


def pair(a, b, pa, pb):
    common = sorted(set(pa) & set(pb))
    if not common:
        raise SystemExit(f"pair {a},{b}: no common problems")
    da = [sum(pa[i]) / len(pa[i]) for i in common]
    db = [sum(pb[i]) / len(pb[i]) for i in common]
    diffs = [y - x for x, y in zip(da, db)]
    m, se = mean_se(diffs)
    out = {"a": a, "b": b, "problems": len(common), "diff_points": round(100 * m, 2), "se_points": round(100 * se, 2),
           "ci95": [round(100 * (m - 1.96 * se), 2), round(100 * (m + 1.96 * se), 2)],
           "a_better": sum(d < 0 for d in diffs), "b_better": sum(d > 0 for d in diffs)}
    if all(len(pa[i]) == 1 and len(pb[i]) == 1 for i in common):
        out["mcnemar_exact_p"] = float(f"{mcnemar_exact(out['a_better'], out['b_better']):.3g}")
    return out


def main():
    self_test()
    args = sys.argv[1:]
    out_json = rows_dir = ids = None
    pairs, folders = [], []
    i = 0
    while i < len(args):
        if args[i] == "--out":
            out_json = args[i + 1]; i += 2
        elif args[i] == "--rows":
            rows_dir = args[i + 1]; i += 2
        elif args[i] == "--ids":
            ids = {x.strip() for x in open(args[i + 1]) if x.strip()}; idfile = args[i + 1]; i += 2
        elif args[i] == "--pair":
            pairs.append(tuple(args[i + 1].split(","))); i += 2
        else:
            folders.append(tuple(args[i].split("=", 1))); i += 1
    if not out_json or not folders:
        raise SystemExit(__doc__)
    res, pers = [], {}
    for label, d in folders:
        s, per = score_dir(label, d, ids, rows_dir)
        res.append(s)
        pers[label] = per
    out = {"scorer": {"python": sys.version.split()[0], "sympy": G._sympy.__version__,
                      "grader_sha256": sha(HERE / "grader.py"), "score_sha256": sha(HERE / "score.py"),
                      "ids_file": Path(idfile).name if ids is not None else None, "n_ids": len(ids) if ids is not None else None},
           "folders": res, "pairs": [pair(a, b, pers[a], pers[b]) for a, b in pairs]}
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
