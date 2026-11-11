"""Cross-setting comparisons on a common set of problems, from score.py --rows output.

Usage: python -I cross_settings.py ROWS_DIR IDS_FILE OUT_JSON [SPLIT]
ROWS_DIR holds base_C3.jsonl, edited_C3.jsonl, base_C2.jsonl, ... (one line per generation, as written by
score.py --rows; a cell may be split over several files whose names start with the label, e.g. edited_C1.jsonl and
edited_C1_next.jsonl). IDS_FILE lists the problems, one id per line. SPLIT (optional) is a line number: the comparisons
are also given for the problems before it and from it on.

Per problem the samples are averaged first (C3 has 4, C1 and C2 have 1), so the standard error is over problems.
The interval is difference +- 1.96 se, unadjusted, and does not allow for choosing the cells after seeing them.
"""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

CELLS = ["base_C3", "edited_C3", "base_C2", "edited_C2", "base_C1", "edited_C1"]
PAIRS = [("edited_C3", "base_C3"), ("edited_C2", "base_C3"), ("edited_C1", "base_C3"),
         ("edited_C1", "edited_C3"), ("edited_C2", "edited_C3"),
         ("edited_C2", "base_C2"), ("edited_C1", "base_C1")]


def load(rows_dir, label):
    per = defaultdict(list)
    for f in sorted(glob.glob(str(Path(rows_dir) / f"{label}*.jsonl"))):
        if Path(f).stem not in (label, f"{label}_next"):
            continue
        for line in open(f):
            r = json.loads(line)
            per[r["id"]].append(1.0 if r["correct"] else 0.0)
    return {k: sum(v) / len(v) for k, v in per.items()}


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def block(cells, ids):
    out = {"problems": len(ids), "cells": {}, "pairs": []}
    for c in CELLS:
        m, se = mean_se([cells[c][i] for i in ids])
        out["cells"][c] = {"accuracy": round(100 * m, 2), "se": round(100 * se, 2)}
    for a, b in PAIRS:
        m, se = mean_se([cells[a][i] - cells[b][i] for i in ids])
        out["pairs"].append({"comparison": f"{a} - {b}", "points": round(100 * m, 2), "se": round(100 * se, 2),
                             "ci95": [round(100 * (m - 1.96 * se), 2), round(100 * (m + 1.96 * se), 2)]})
    return out


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    rows_dir, ids_file, out_json = sys.argv[1:4]
    ids = [x.strip() for x in open(ids_file) if x.strip()]
    cells = {c: load(rows_dir, c) for c in CELLS}
    for c in CELLS:
        missing = [i for i in ids if i not in cells[c]]
        if missing:
            raise SystemExit(f"{c}: {len(missing)} problems missing, e.g. {missing[0]}")
    out = {"all": block(cells, ids)}
    if len(sys.argv) == 5:
        k = int(sys.argv[4])
        out["first"] = block(cells, ids[:k])
        out["rest"] = block(cells, ids[k:])
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
