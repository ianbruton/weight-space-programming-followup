# Part 1: MATH-500 under three settings, the grader and the scoring

| Script | Needs | What |
|---|---|---|
| `gen_job.py` | GPU, vLLM 0.31.0 | One model, one setting (C1, C2 or C3), a list of problems. Seed of sample k of problem i is `1000·i + k`, the same for every model. Writes `chunk_<j>.jsonl.gz` (id, k, tokens, finish reason, text) and `meta.json` (timings). Resumable by chunk. `--stop-by EPOCH` stops before a chunk that would run past a deadline. |
| `grader.py` | sympy | The grader. Extraction follows the authors' rule: the last `\boxed{}` after the last `</think>`, or in the whole text if there is none. Equivalence: MATH-style normalisation, then tuples and lists element-wise, numbers through sympy, and expressions in variables symbolically. A trailing `\text{}` holding only unit words is dropped. |
| `score.py DIR ... [--json OUT]` | `MATH500` env var | Accuracy with its standard error over problems, share at the token cap, mean and median tokens, loops among capped answers. Pairs folders named `base_<x>` / `edited_<x>` and gives the paired difference with its standard error. |
| `check_grader.py GRADER_DIR LOGS_EXP8_DIR MATH500_JSONL` | CPU | Compares `grader.equiv` with the authors' math-verify flags on their released first answers. Ours: 338 agree, 0 disagree. |
| `regrade.py CARD_EVAL_DIR LOGS_EXP8_DIR TOKENIZER_DIR OUT_JSON` | transformers (tokenizer only) | Re-grades the released texts (the model card's re-run and 20 texts per shard of the authors' runs) with an earlier, simpler grader, and re-sums the authors' flags and cap shares. |

The settings:

| | thinking | temperature | top_p | top_k | presence penalty | max new tokens |
|---|---|---|---|---|---|---|
| C1, the post's | on | 0.6 | 0.95 | 20 | 1.5 | 32,768 |
| C2, the card's thinking | on | 1.0 | 0.95 | 20 | 1.5 | 81,920 |
| C3, the card's non-thinking | off | 1.0 | 1.0 | 20 | 2.0 | 32,768 |

The full command sequence for one RTX 3090 is in the top-level README. C1 and C2 use one sample on the 100 problems in
`data/part1/order100.txt`; C3 uses four samples on all 500.
