# The harness essentials: Astra's conditions for a weight editor

The README of MIT's maths experiment (`exp8_math/README.md` in the authors' logs) lists the agent harness code (the
prompts, sandbox and forward-pass server) among what is not included. We rebuilt its conditions from the run's prompt
in the logs (`exp8_math/runs/p50c/turns/t001/prompt.txt`). These are the parts that define what an
editor can do; the host driver that queued jobs on our GPU and sandboxed the kernel is machine-specific and is not
published.

| File | What |
|---|---|
| `kernel_server.py` | A persistent Python kernel driven by files: it runs `inbox/<seq>.py` in one namespace that survives between runs and writes `outbox/<seq>.out`. If the last statement is an expression, its value is shown, as in a notebook. |
| `prelude.py` | Run once at kernel start. Defines what Astra's kernel had: `NAMES`, `load(name)`, `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()` and `tok`. Gradients are off. |
| `rules.py` | The edit rules, checked by `save_edit` and again on the host before every edited forward pass: refused are the final norm, the embedding / output layer, any edit that only multiplies a whole tensor by a constant, names outside the text model, wrong shapes and non-finite values. |
| `fwd_job.py PASS_DIR` | One forward pass: either a generation through vLLM (original, edited or both models, with shared per-item seeds), or a teacher-forced read through transformers that returns log-probabilities, hidden states and MLP activations for chosen layers. The request limits (Astra's: up to 512 answers, or 1,024 texts) were checked by the host driver before a pass was queued. `WSP_FAKE=1` runs it without a GPU, for tests. |

The conditions we matched: the kernel has no network and no GPU, 900 s per run, output cut at 10,000 characters, no
training (no gradients, optimisers or iterative fitting), a fixed budget of forward passes, and failed requests not
counted. What we could not match: our kernel had 24 GB per run (Astra's 96 GB), and no scipy, scikit-learn or pandas.

Sandboxing: run the kernel in a container or namespace with no network and no GPU device, the model folder mounted
read-only at `/model`, practice files read-only at `/practice`, this folder read-only at `/harness`, and a writable
`/work`.
