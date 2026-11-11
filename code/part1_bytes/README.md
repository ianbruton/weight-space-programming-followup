# Part 1: the bytes of the edit

CPU only. Needs `torch`, `numpy`, `safetensors`.

| Script | What |
|---|---|
| `bytes_diff.py BASE EDITED OUT_DIR` | Compares the two single-file checkpoints tensor by tensor on their raw bf16 bits. Lists the tensors that differ, counts the changed entries, rows and columns, and hashes each tensor. Fits the change in each edited tensor as `ΔW = −k·u(uᵀW0)` (SVD, then a closed-form refinement) and reports `k` per column. Writes `OUT_DIR/bytes_result.json` and `OUT_DIR/u_layer{15,19}.npy`. |
| `u_exact.py BASE EDITED OUT_DIR` | Starting from those `u`, searches for one `u` per layer that rebuilds every edited bf16 entry from `W0 − 6·u(uᵀW0)`. Each bf16 value fixes an interval of reals that round to it; the search alternates between `p = uᵀW0` and interval midpoints. Writes `u_layer{15,19}_refined.npy` and `u_exact_result.json`. |

Inputs: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` and
`RikiyaT/Qwen3.5-0.8B-weight-space-programming` at `d1671a178705bcab7644806d23c535517f35723b`.

Expected (our run, and the published `data/part1/bytes.json`): 2 of 488 tensors differ, 130,188 entries,
`k` = 6.0006 ± 0.0018 and 6.0004 ± 0.0021, cos(u15, u19) = 0.849, and 133 / 118 of 65,536 entries per tensor not
rebuilt bit-exactly. The refined directions are published in `data/part1/directions_0p8b.json`.
