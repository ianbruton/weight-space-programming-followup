# Part 2: finding and writing a weight edit on Qwen3.5-4B

These are the scripts of our 4B citation edit, with machine-specific paths removed. The benchmark they were run on is
private, so its data are not included. The scripts are published as method: give them inputs of the same shape and
they run. Set `QWEN4B_DIR` to `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

| Script | Needs | What |
|---|---|---|
| `lib.py` | | Shared pieces: the held-out split (`sha256(key) mod 5 < 2`), forward hooks that keep each block's output, each MLP's output and each MLP's inner activation, a teacher-forced forward pass, and `apply_edit` (the edit in memory, with the weight files' arithmetic). |
| `c_read.py FINAL_JSONL OUT_DIR` | GPU (about 19 GB peak) | Teacher-forced read of the base model over every recorded answer. Per citation and per layer it keeps the residual stream, the MLP output and the MLP inner activation, averaged over four token sets (`pre`, `ref`, `quote`, `item`). For each wrong-path miss it also reads a second branch with a path that holds. |
| `d_build.py READ_DIR OUT.json SET CAND K [LAYERS] [SEED]` | CPU | Builds one edit family with Astra's rules: a per-layer difference of means on practice items (`CAND`: `natural`, `matched` or `counterfactual`), the layers whose MLP writes most of it, the top `K` columns by `(uᵀw_j) × activation contrast`, and a norm-matched random control. Our F1 was `item matched 165 29,31`. |
| `f_dose.py FINAL_JSONL FAMILY.json OUT.json [practice]` | GPU | Teacher-forced dose screen: for the base and each dose of the target and control, the change at each fork in `log p(right path token) − log p(wrong path token)`, and the cost on held citations. |
| `e_write.py EDIT.json OUT_DIR` | CPU, low memory | Writes an edited copy as a new model folder. Every unchanged file is a symlink to the base; the one shard holding the edited tensors is copied, and only those tensors' bytes are overwritten. Checks byte range by byte range that nothing else changed, and prints a manifest hash of the folder. |
| `n_boot.py ARM=RUN_DIR ... -- A:B ...` | CPU | Pooled share of citations that hold per arm, and a bootstrap over items (10,000 draws, the same draw for both arms, seed 20261110) for each difference. Reads each run's `proposals.jsonl`. |

**Input shape of `FINAL_JSONL`** (one line per recorded final answer):
`{"run", "key", "kind", "prompt", "prompt_tokens_vllm", "answer", "items": [...]}`, where each item is one citation:
`{"index", "item": {"source", "ref", "quote"}, "held": bool, "why", "ref": [start, end], "quote": [start, end],
"item_span": [start, end], "paths_holding": [...]}`. The spans are character offsets into `answer` (the `ref` and
`quote` spans include their quote marks). `paths_holding` lists paths that would have held for that quote; it is empty
for citations that held.

**The edit itself** (no weights needed): `data/part2/direction_4b_F1.json` holds `u` and the 165 columns for layers
29 and 31, and the control. Apply `W ← W0 − dose·alpha_scale·u(uᵀW0)` on those columns of
`model.language_model.layers.<l>.mlp.down_proj.weight` (float32 from bf16, cast back). Dose 1 is the edit we tested.
`e_write.py` does exactly this given `{"alpha": 1, "layers": {"29": {"u": [...], "columns": [...]}, "31": {...}}}`.
