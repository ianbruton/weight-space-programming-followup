"""One model, one sampling setting, a list of math items, through vLLM offline (one GPU job).

Usage: <venv python> -I gen_job.py --model DIR --out DIR --items JSONL --setting C1|C2|C3 [--n N] [--max-tokens T]
         [--ids FILE] [--chunk K] [--stop-by EPOCH] [--first-est S] [--gpu-mem F]
Items: one JSON object per line with id (or unique_id), problem, answer. The user turn is the problem, a newline and
"Please reason step by step, and put your final answer within \\boxed{}." (the evaluation's prompt), through the
model's own chat template. Settings (sampling):
  C1 the post's:            thinking, T 0.6, top_p 0.95, top_k 20, min_p 0, presence 1.5, max 32,768
  C2 card, thinking:        thinking, T 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5, max 81,920
  C3 card, non-thinking:    no thinking, T 1.0, top_p 1.0, top_k 20, min_p 0, presence 2.0, max 32,768
Seed of sample k of item i: 1000 * i + k (fixed by the item, the same for every model).
Writes OUT/chunk_<j>.jsonl.gz (id, k, n_tokens, finish, text) and OUT/meta.json (timings, tokens). Resumable: chunks
already written are skipped. Stops before a chunk that would run past --stop-by (exit 0, meta says so).
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

SETTINGS = {
    "C1": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, max_tokens=32768),
    "C2": dict(thinking=True, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, max_tokens=81920),
    "C3": dict(thinking=False, temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, max_tokens=32768),
}
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--stop-by", type=float, default=0)
    ap.add_argument("--first-est", type=float, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--only-chunks", help="comma-separated chunk numbers to run (others are left for later)")
    a = ap.parse_args()

    if a.setting == "C2":
        # measured on one RTX 3090: a C2 chunk of 50 answers on the base 0.8B, 83% of them at the 81,920 cap,
        # took 3,163 and 3,213 s (17.5 answers fit in the KV cache at full length); never start one with less than
        # 3,600 s before the stop
        a.first_est = max(a.first_est, 3600)
    if a.setting == "C1":
        a.first_est = max(a.first_est, 1800)   # thinking at 32,768 tokens: most base answers run to the cap
    st = dict(SETTINGS[a.setting])
    thinking = st.pop("thinking")
    if a.max_tokens:
        st["max_tokens"] = a.max_tokens
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(a.items)]
    for i, r in enumerate(rows):
        r["_i"] = i
        r.setdefault("id", r.get("unique_id"))
    if a.ids:
        want = [x.strip() for x in open(a.ids) if x.strip()]
        by = {r["id"]: r for r in rows}
        rows = [by[x] for x in want]
    chunks = [rows[j:j + a.chunk] for j in range(0, len(rows), a.chunk)]
    mp = out / "meta.json"
    meta = json.load(open(mp)) if mp.exists() else {}
    job = {"model": a.model, "items": a.items, "setting": a.setting, "sampling": st, "thinking": thinking, "n": a.n,
           "ids": a.ids, "chunk": a.chunk, "n_items": len(rows)}
    if meta.get("job") and meta["job"] != job:
        raise SystemExit(f"{out} holds another job: {meta['job']}")
    meta["job"] = job
    meta.setdefault("chunks", {})
    meta.setdefault("runs", [])
    todo = [j for j in range(len(chunks)) if not (out / f"chunk_{j}.jsonl.gz").exists()]
    if a.only_chunks:
        keep = {int(x) for x in a.only_chunks.split(",")}
        todo = [j for j in todo if j in keep]
    run = {"started": time.time(), "todo": todo, "stopped_for_deadline": False}
    meta["runs"].append(run)


    def save_meta():
        tmp = mp.with_suffix(".tmp")
        json.dump(meta, open(tmp, "w"), indent=1)
        os.replace(tmp, mp)


    save_meta()
    if not todo:
        return

    if a.stop_by and time.time() + a.first_est > a.stop_by:      # do not even load the model
        run["stopped_for_deadline"] = True
        run["stopped_before_chunk"] = todo[0]
        save_meta()
        return
    from vllm import LLM, SamplingParams  # noqa: E402

    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=st["max_tokens"] + 2048, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=False, language_model_only=True, attention_backend="TRITON_ATTN", seed=0)
    tok = llm.get_tokenizer()
    run["load_s"] = round(time.time() - t0, 1)
    save_meta()
    est = a.first_est
    for j in todo:
        if a.stop_by and time.time() + est > a.stop_by:
            run["stopped_for_deadline"] = True
            run["stopped_before_chunk"] = j
            break
        ch = chunks[j]
        prompts, sps, keys = [], [], []
        for r in ch:
            p = tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=thinking)
            for k in range(a.n):
                prompts.append(p)
                sps.append(SamplingParams(seed=1000 * r["_i"] + k, repetition_penalty=1.0, **st))
                keys.append((r["id"], k))
        t1 = time.time()
        res = llm.generate(prompts, sps)
        dt = time.time() - t1
        bad = sum(1 for o in res if o.outputs[0].finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations neither stopped nor reached max_tokens; chunk {j} not saved")
        ntok = 0
        tmp = out / f"chunk_{j}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt") as f:
            for (rid, k), o in zip(keys, res):
                c = o.outputs[0]
                ntok += len(c.token_ids)
                f.write(json.dumps({"id": rid, "k": k, "n_tokens": len(c.token_ids), "finish": c.finish_reason,
                                    "prompt_tokens": len(o.prompt_token_ids), "text": c.text}) + "\n")
        os.replace(tmp, out / f"chunk_{j}.jsonl.gz")
        meta["chunks"][str(j)] = {"gens": len(keys), "seconds": round(dt, 1), "tokens": ntok,
                                  "tok_per_s": round(ntok / dt), "finished_at": time.time()}
        save_meta()
        print(f"chunk {j}: {len(keys)} gens, {ntok} tokens, {dt:.0f} s, {ntok / dt:.0f} tok/s", flush=True)
        est = dt
    run["ended"] = time.time()
    save_meta()


if __name__ == "__main__":
    main()
