# Part 1: MATH-500 under three settings, the grader and the scoring

| Script | Needs | What |
|---|---|---|
| `gen_job.py` | GPU, vLLM 0.31.0 | One model, one setting (C1, C2 or C3), a list of problems. Seed of sample k of problem i is `1000·i + k`, the same for every model. Writes `chunk_<j>.jsonl.gz` (id, k, tokens, finish reason, text) and `meta.json` (timings). Resumable by chunk. `--stop-by EPOCH` stops before a chunk that would run past a deadline. |
| `grader.py` | sympy 1.14.0 | The pinned grader. Extraction follows the authors' rule: the last `\boxed{}` after the last `</think>`, or in the whole text if there is none. Equivalence: MATH-style normalisation, then tuples and lists element-wise, numbers through sympy, and expressions in variables symbolically. A trailing `\text{}` holding only unit words is dropped. It imports SymPy when it loads and stops unless it is version 1.14.0: the first version caught a missing SymPy silently and graded `10/11` against `\frac{10}{11}` as wrong. A base subscript is dropped only when one side has none (`52_8` = `52`, but `52_9` ≠ `52_8`). A supplementary extraction (a plain final "Answer: X" line) is kept apart for counting. |
| `score.py --out OUT.json [--rows DIR] [--ids FILE] LABEL=DIR ... [--pair A,B ...]` | `MATH500` env var, sympy 1.14.0 | Accuracy with its standard error over problems (samples of a problem averaged first), with the supplementary extraction apart, share at the token cap, mean tokens, loops among capped answers, finished answers with no box. Per pair: the difference, its standard error, an approximate 95% interval, discordant problems and, for one-sample pairs, the exact McNemar p. Refuses to start unless the SymPy path is live. `--rows` writes one line per generation. |
| `check_grader.py LOGS_EXP8_DIR MATH500_JSONL` | sympy 1.14.0 | Exits 1 on any disagreement. (A) `grader.equiv` against the authors' math-verify flags on their released first answers: 338 agree, 0 disagree; (B) extraction against the authors' extracted answer on 400 released texts: 400 agree; (C) 17 unit cases. |
| `cross_settings.py ROWS_DIR IDS_FILE OUT_JSON [SPLIT]` | CPU | From `score.py --rows` output: all six cells on a common set of problems and the cross-setting differences (edited C1 − base C3 and so on), optionally split at a line of the id file. |
| `regrade.py CARD_EVAL_DIR LOGS_EXP8_DIR TOKENIZER_DIR OUT_JSON` | transformers (tokenizer only) | Re-grades the released texts (the model card's re-run and 20 texts per shard of the authors' runs) with an earlier, simpler grader, and re-sums the authors' flags and cap shares. |

The settings:

| | thinking | temperature | top_p | top_k | presence penalty | max new tokens |
|---|---|---|---|---|---|---|
| C1, the post's | on | 0.6 | 0.95 | 20 | 1.5 | 32,768 |
| C2, the card's thinking | on | 1.0 | 0.95 | 20 | 1.5 | 81,920 |
| C3, the card's non-thinking | off | 1.0 | 1.0 | 20 | 2.0 | 32,768 |

The full command sequence for one RTX 3090 is in the top-level README. C1 and C2 use one sample on the 200 problems in
`data/part1/order200.txt` (its first 100 are `order100.txt`); C3 uses four samples on all 500.
