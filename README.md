# Weight-space programming: a follow-up

MIT CSAIL's [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab, November 2026) showed a frontier model editing Qwen3.5-0.8B's weights directly, with no
training, and lifting MATH-500 from 7.9% to 48.1%. This repository holds a follow-up in three parts:

1. **MIT's 0.8B edit, reproduced and compared fairly.** The bytes of the release, the dose of 6, both models under
   three sampling settings, compared across settings, and what the edit costs (perplexity, tool use, an agent task).
2. **Our own 4B citation edit.** The same recipe on Qwen3.5-4B for a different failure, with a random-direction
   control and statistics reported as planned and as extended, then a second round after review.
3. **A 27B tool-reflex edit.** A negative result: the reflex turned out to be the model's ordinary checking habit,
   and the edit the recipe found moved the call instead of removing it.

All three parts were reviewed by GPT-6 Astra as a strict reviewer; the corrections are marked on each page.
Live site: https://ianbruton.github.io/weight-space-programming-followup/

The write-up is a static site in `docs/` (plain HTML and CSS, no build step, no scripts). Open `docs/index.html` in a
browser, or publish `docs/` with GitHub Pages.

## Layout

| Path | What |
|---|---|
| `docs/` | the site: `index.html`, `part1.html`, `part2.html`, `part3.html`, `data.html`, `style.css` |
| `docs/data/` (also `data/`, a link to it) | the result numbers as JSON and CSV, aggregates only |
| `code/part1_bytes/` | byte comparison of the two checkpoints and recovery of the edit direction |
| `code/part1_math500/` | generation under the three settings, the grader, the scoring, the re-grade of released texts |
| `code/part2_edit_4b/` | reading the 4B, building the direction, writing an edited copy, the dose screen, the bootstrap |
| `code/harness/` | the essentials of a weight-editing harness in Astra's conditions (kernel, edit rules, forward pass) |

Each `code/` folder has its own README.

## Reproduce Part 1 on one RTX 3090

Everything in Part 1 uses public files. Our runs used vLLM 0.31.0 in bf16 on one RTX 3090 (24 GB).

```sh
# 1. Environment (uv)
uv venv .venv && . .venv/bin/activate
uv pip install "vllm==0.31.0" safetensors "sympy==1.14.0" numpy huggingface_hub

# 2. Inputs, pinned to the revisions we and the independent audit used
hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/base
hf download RikiyaT/Qwen3.5-0.8B-weight-space-programming \
  --revision d1671a178705bcab7644806d23c535517f35723b --local-dir models/edited
hf download HuggingFaceH4/MATH-500 test.jsonl --repo-type dataset \
  --revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be --local-dir math500
git clone https://github.com/RikiyaT/lm_understand-logs logs && git -C logs checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

# 3. The bytes (CPU, a few minutes)
B=models/base/model.safetensors-00001-of-00001.safetensors
E=models/edited/model.safetensors-00001-of-00001.safetensors
python -I code/part1_bytes/bytes_diff.py $B $E out/bytes
python -I code/part1_bytes/u_exact.py $B $E out/bytes

# 4. Re-grade the released texts (CPU)
python -I code/part1_math500/regrade.py models/edited/eval logs/exp8_math models/base out/regrade_result.json

# 5. Generate (GPU). C3: 4 samples on all 500; C2 and C1: 1 sample on the same 200 problems
G="python -I code/part1_math500/gen_job.py --items math500/test.jsonl"
$G --model models/base   --setting C3 --n 4 --chunk 250 --out out/base_C3
$G --model models/edited --setting C3 --n 4 --chunk 250 --out out/edited_C3
$G --model models/base   --setting C2 --n 1 --chunk 50 --ids data/part1/order200.txt --out out/base_C2x1
$G --model models/edited --setting C2 --n 1 --chunk 50 --ids data/part1/order200.txt --out out/edited_C2x1
$G --model models/base   --setting C1 --n 1 --chunk 50 --ids data/part1/order200.txt --out out/base_C1x1
$G --model models/edited --setting C1 --n 1 --chunk 50 --ids data/part1/order200.txt --out out/edited_C1x1

# 6. Score (CPU) with the pinned scorer (it stops if sympy 1.14.0 is missing)
python -I code/part1_math500/check_grader.py logs/exp8_math math500/test.jsonl   # must exit 0
MATH500=math500/test.jsonl python -I code/part1_math500/score.py --out out/scores.json --rows out/rows \
  base_C3=out/base_C3 edited_C3=out/edited_C3 base_C2=out/base_C2x1 edited_C2=out/edited_C2x1 \
  base_C1=out/base_C1x1 edited_C1=out/edited_C1x1 \
  --pair base_C3,edited_C3 --pair base_C2,edited_C2 --pair base_C1,edited_C1
python -I code/part1_math500/cross_settings.py out/rows data/part1/order200.txt out/cross200.json 100
```

Generation times on our GPU: C3 about 69 and 64 minutes (base, edited) for 2,000 answers each; C2 199 and 109 minutes
for 200 answers; C1 42 and 28 minutes. Seeds are fixed per problem and sample (1000·i + k), so the two models see the
same seeds. Sampled runs on another GPU or vLLM version will not match ours token for token. Expect the same picture
within the standard errors in `data/part1/math500_settings.csv`.

Parts 2 and 3 use a private benchmark. Their code is published as method (see `code/part2_edit_4b/README.md`), and
their results as aggregates.

## What is not here, and why

- **No model weights.** The edits are published as direction vectors and column lists with the formula that applies
  them (`data/part1/directions_0p8b.json`, `data/part2/direction_4b_F1.json`).
- **No private data.** The agent benchmark in Parts 1–3 is private. We describe its shape and publish aggregates
  only. The perplexity texts are private files, so we give counts.
- **No copy of MIT's code or logs.** They are linked. Their logs are CC BY 4.0 and their scripts MIT; the numbers we
  quote from them are attributed on each page.

## Licence

Code (`code/`): MIT, see `LICENSE`. The site's text and the data files (`docs/`): CC BY 4.0, see `LICENSE-CONTENT`.
Third-party material keeps its own licence (MIT CSAIL's logs CC BY 4.0; the datasets theirs).

## Credits

- Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL): the post, the edit, the
  [logs](https://github.com/RikiyaT/lm_understand-logs) (CC BY 4.0, scripts MIT) and the
  [edited model](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) (Apache 2.0).
- Carlos Toledo: the [independent audit](https://carlostoledo.co/reports/weight-space-programming.html), which first
  decided the edit from its bytes and ran the unedited model under the model card's settings.
- Ahsen Khaliq (Hugging Face): the rebuild and re-run on the model card.
- The Qwen team (Qwen3.5), HuggingFaceH4 (MATH-500), and the authors of the Hermes function-calling, ToolACE and
  Glaive function-calling datasets.

The experiments were designed and run by Claude agent sessions in Claude Code under the author's direction, on one
RTX 3090.
