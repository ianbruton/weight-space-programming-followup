"""One forward pass of the harness (the body of one GPU job). The host driver (not published) prepared PASS_DIR:
  request.json  the validated request (normalised), with 'models': {label: model_dir}
  items.jsonl   generation: one line per generation {key, prompt, seed}; teacher-forced: one line per text {text}
Usage: <venv python> -I fwd_job.py PASS_DIR            (orchestrates; each step is its own process, so the GPU is
       freed between vLLM and transformers)
       <venv python> -I fwd_job.py PASS_DIR gen LABEL
       <venv python> -I fwd_job.py PASS_DIR read LABEL MODE   (MODE: tf | acts)
Writes PASS_DIR/out/... and PASS_DIR/job_meta.json (seconds per step, tokens). WSP_FAKE=1: no GPU, no model, canned
answers and random arrays of the right shapes (for the harness's tests).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

FAKE = os.environ.get("WSP_FAKE") == "1"
P = Path(sys.argv[1]) if len(sys.argv) > 1 else None
REQ = json.load(open(P / "request.json")) if P else None
OUT = P / "out" if P else None


def items():
    return [json.loads(line) for line in open(P / "items.jsonl")]


def meta_update(key, val):
    mp = P / "job_meta.json"
    m = json.load(open(mp)) if mp.exists() else {}
    m[key] = val
    json.dump(m, open(mp, "w"), indent=1)


# ------------------------------------------------------------------ generation (vLLM)
def gen(label):
    rows = items()
    t0 = time.time()
    st = REQ["sampling"]
    if FAKE:
        canned = ["Thinking... 2+2=4.</think>\n\nSo \\boxed{4}.", "loop loop loop " * 50, "</think> \\boxed{1}"]
        res = [(canned[i % 3], 10 + i % 7, "length" if i % 3 == 1 else "stop") for i in range(len(rows))]
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=REQ["models"][label], dtype="bfloat16", max_model_len=REQ["max_tokens"] + 2048,
                  gpu_memory_utilization=float(os.environ.get("WSP_GPU_MEM", "0.88")), enable_prefix_caching=False,
                  language_model_only=True, attention_backend="TRITON_ATTN", seed=0)
        t_load = time.time() - t0
        sps = [SamplingParams(seed=r["seed"], max_tokens=REQ["max_tokens"], repetition_penalty=1.0, **st) for r in rows]
        t1 = time.time()
        outs = llm.generate([r["prompt"] for r in rows], sps)
        bad = sum(1 for o in outs if o.outputs[0].finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations neither stopped nor reached max_tokens")
        res = [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]
    with open(OUT / f"gen_{label}.jsonl", "w") as f:
        for r, (text, n, fin) in zip(rows, res):
            f.write(json.dumps({"key": r["key"], "text": text, "n_tokens": n, "finish": fin}) + "\n")
    meta_update(f"gen_{label}", {"seconds": round(time.time() - t0, 1),
                                 "load_seconds": round(t_load, 1) if not FAKE else 0,
                                 "generate_seconds": round(time.time() - t1, 1) if not FAKE else 0,
                                 "tokens": sum(x[1] for x in res), "gens": len(res)})


# ------------------------------------------------------------------ teacher-forced reads (transformers)
def read(label, mode):
    import numpy as np
    t0 = time.time()
    ret = REQ.get("return", {})
    if mode == "acts":       # activations of generated answers: prompt + the answer's first ACT_TOKENS tokens
        gens = [json.loads(line) for line in open(OUT / f"gen_{label}.jsonl")]
        prompts = {r["key"]: r["prompt"] for r in items()}
        texts = [(prompts[g["key"]], g["text"]) for g in gens]
        layers = REQ["activations"]["layers"]
        mlp = REQ["activations"].get("mlp", False)
        ret = {"hidden_states": layers, "mlp_act": layers[:] if mlp else []}
    else:
        texts = [(None, r["text"]) for r in items()]
    hs_layers, mlp_layers = ret.get("hidden_states", []), ret.get("mlp_act", [])
    attn_layers, topk, want_lp = ret.get("attn_layers", []), int(ret.get("topk", 0)), ret.get("token_logprobs", False)
    dest = OUT / (f"acts_{label}" if mode == "acts" else "tf")
    dest.mkdir(exist_ok=True)
    if FAKE:
        d, dm, n_layers = 16, 32, 33
        lens = [min(len(t[1].split()) + 1, 64) for t in texts]
        plens = [3 for _ in texts]
        tok = None
    else:
        import torch
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM  # gitleaks:allow (an import line, not a secret)
        mdir = REQ["models"][label]
        tok = AutoTokenizer.from_pretrained(mdir)
        cfg = AutoConfig.from_pretrained(mdir)
        tcfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
        if attn_layers:
            tcfg._attn_implementation = "eager"
        dev = os.environ.get("WSP_DEVICE", "cuda")
        model = Qwen3_5ForCausalLM.from_pretrained(mdir, config=tcfg, dtype=torch.bfloat16,
                                                   key_mapping={r"^model\.language_model\.": "model."})
        model.to(dev).eval()
        d, dm, n_layers = tcfg.hidden_size, tcfg.intermediate_size, tcfg.num_hidden_layers + 1
        enc, plens = [], []
        for pr, tx in texts:
            if pr is None:
                ids = tok(tx, add_special_tokens=False)["input_ids"]
                plens.append(0)
            else:
                p_ids = tok(pr, add_special_tokens=False)["input_ids"]
                a_ids = tok(tx, add_special_tokens=False)["input_ids"][:REQ["act_tokens"]]
                ids = p_ids + a_ids
                plens.append(len(p_ids))
            enc.append(ids)
        lens = [len(e) for e in enc]
    n, T = len(texts), max(lens)
    acts_mode = mode == "acts"
    # outputs: tf -> right-padded [n, T, ...]; acts -> per sample [n, L, d] at the prompt's last token and averaged
    store = {}
    for layer in hs_layers:
        store[f"hs_{layer}"] = np.zeros((n, 2, d) if acts_mode else (n, T, d), np.float16 if not acts_mode else np.float32)
    for layer in mlp_layers:
        store[f"mlp_{layer}"] = np.zeros((n, 2, dm) if acts_mode else (n, T, dm),
                                         np.float16 if not acts_mode else np.float32)
    if want_lp:
        store["token_logprobs"] = np.full((n, T), np.nan, np.float32)
    if topk:
        store["topk_ids"] = np.zeros((n, T, topk), np.int32)
        store["topk_logprobs"] = np.zeros((n, T, topk), np.float32)
    attn = {}
    if FAKE:
        rng = np.random.default_rng(0)
        for k, a in store.items():
            a[...] = rng.standard_normal(a.shape) if a.dtype != np.int32 else 1
    else:
        layers_mod = model.model.layers
        cap = {}
        hooks = []
        if 0 in hs_layers:
            hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("hs_0", o)))
        for layer in hs_layers:
            if layer > 0:
                hooks.append(layers_mod[layer - 1].register_forward_hook(
                    lambda m, i, o, layer=layer: cap.__setitem__(f"hs_{layer}", o[0] if isinstance(o, tuple) else o)))
        for layer in mlp_layers:
            hooks.append(layers_mod[layer].mlp.down_proj.register_forward_pre_hook(
                lambda m, i, layer=layer: cap.__setitem__(f"mlp_{layer}", i[0])))
        for layer in attn_layers:
            hooks.append(layers_mod[layer].self_attn.register_forward_hook(
                lambda m, i, o, layer=layer: cap.__setitem__(f"attn_{layer}", o[1])))
        with torch.no_grad():
            for b in range(n):     # one text at a time: exact, no padding inside the model
                ids = torch.tensor([enc[b]], device=dev)
                cap.clear()
                hsd = model.model(input_ids=ids).last_hidden_state[0]      # after the final norm
                L, pl = lens[b], plens[b]
                for k in list(store):
                    if k.startswith(("hs_", "mlp_")):
                        x = cap[k][0].float()
                        if acts_mode:
                            store[k][b, 0] = x[pl - 1].cpu().numpy()
                            store[k][b, 1] = (x[pl:].mean(0) if L > pl else x[pl - 1]).cpu().numpy()
                        else:
                            store[k][b, :L] = x.cpu().numpy().astype(np.float16)
                for layer in attn_layers:
                    attn.setdefault(layer, []).append(cap[f"attn_{layer}"][0].float().cpu().numpy().astype(np.float16))
                if want_lp or topk:
                    for s in range(0, L, 1024):
                        e = min(L, s + 1024)
                        lp = torch.log_softmax(model.lm_head(hsd[s:e]).float(), -1)   # predicts token t+1
                        if want_lp:
                            nxt = ids[0, s + 1:e + 1]
                            m = nxt.shape[0]
                            store["token_logprobs"][b, s + 1:s + 1 + m] = lp[:m].gather(1, nxt[:, None])[:, 0].cpu().numpy()
                        if topk:
                            v, ix = lp.topk(topk, -1)
                            store["topk_logprobs"][b, s:e] = v.cpu().numpy()
                            store["topk_ids"][b, s:e] = ix.cpu().numpy()
        for h in hooks:
            h.remove()
    for k, a in store.items():
        np.save(dest / f"{k}.npy", a)
    for layer, arrs in attn.items():
        for b, a in enumerate(arrs):
            np.save(dest / f"attn_{layer}_text{b}.npy", a)
    info = {"n_texts": n, "lengths": lens, "prompt_lengths": plens, "layers_hidden": hs_layers, "layers_mlp": mlp_layers,
            "attn_layers": attn_layers, "files": sorted(os.listdir(dest)),
            "layout": ("per sample: [:, 0] the prompt's last token, [:, 1] the mean over the answer's first "
                       f"{REQ.get('act_tokens')} tokens; hs_l = residual after layer l (0 = embeddings); mlp_l = "
                       "layer l's down_proj input") if acts_mode else
                      ("right-padded [text, position, ...]; token_logprobs[t] = log p(token t | tokens < t) "
                       "(NaN at 0 and on padding); topk_* at position t = the next-token distribution after token t; "
                       "hs_l = residual after layer l (0 = embeddings), float16; mlp_l = layer l's down_proj input, "
                       "float16")}
    if tok is not None and not acts_mode:
        info["token_ids"] = enc
    json.dump(info, open(dest / "meta.json", "w"))
    meta_update(f"read_{mode}_{label}", {"seconds": round(time.time() - t0, 1), "tokens": int(sum(lens))})


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 2:
        if sys.argv[2] == "gen":
            gen(sys.argv[3])
        else:
            read(sys.argv[3], sys.argv[4])
        return

    # orchestrate
    if os.environ.get("WSP_FAKE_FAIL") == "1":
        meta_update("failed_step", "WSP_FAKE_FAIL")
        sys.exit(1)
    t0 = time.time()
    steps = []
    if REQ["type"] == "generate":
        for label in REQ["models"]:
            steps.append(["gen", label])
            if REQ.get("activations"):
                steps.append(["read", label, "acts"])
    else:
        steps.append(["read", next(iter(REQ["models"])), "tf"])
    for s in steps:
        r = subprocess.run([sys.executable, "-I", __file__, str(P), *s])
        if r.returncode:
            meta_update("failed_step", s)
            sys.exit(r.returncode)
    meta_update("total_seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
