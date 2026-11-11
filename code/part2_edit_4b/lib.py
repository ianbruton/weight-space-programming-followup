"""Shared pieces for the 4B reading and editing scripts (transformers; the base
files are opened read-only). Set QWEN4B_DIR to the folder of
Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a."""

import hashlib
import json
import os
from pathlib import Path

import torch

MODEL = Path(os.environ.get("QWEN4B_DIR", "models/Qwen3.5-4B"))
N_LAYERS, HIDDEN, INTER = 32, 2560, 9216
SETS = ("pre", "ref", "quote", "item")


def split_of(key: str) -> str:
    """Held-out split by finding key, fixed before any internal state was read."""
    return (
        "heldout"
        if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 < 2
        else "practice"
    )


def load(model_dir=MODEL, device="cuda"):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # gitleaks:allow (an import line, not a secret)

    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=True
    )
    model.eval().to(device)
    return tok, model


def layers_of(model):
    return model.model.language_model.layers


class Taps:
    """Forward hooks that keep, from position `start` on, each block's output,
    each MLP's output and each MLP's inner activation (down_proj's input)."""

    def __init__(self, model, acts=True):
        self.start = 0
        self.h, self.m, self.a = {}, {}, {}
        self.handles = []
        for i, layer in enumerate(layers_of(model)):
            self.handles.append(layer.register_forward_hook(self._out(self.h, i + 1)))
            self.handles.append(layer.mlp.register_forward_hook(self._out(self.m, i)))
            if acts:
                self.handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(self._pre(self.a, i))
                )
        self.handles.append(
            layers_of(model)[0].register_forward_pre_hook(
                self._first, with_kwargs=True
            )
        )

    def _out(self, store, i):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            store[i] = out[0, self.start :].detach().clone()

        return hook

    def _pre(self, store, i):
        def hook(module, args):
            store[i] = args[0][0, self.start :].detach().clone()

        return hook

    def _first(self, module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        self.h[0] = x[0, self.start :].detach().clone()

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward(model, ids, start, taps=None):
    """Run the language model over ids; return the log-probabilities the model
    gives to ids[start:] (each from the position before it). taps.start is set
    to start - 1 so that row r of every tap is sequence position start - 1 + r."""
    assert start >= 1
    if taps is not None:
        taps.start = start - 1
        taps.h.clear(), taps.m.clear(), taps.a.clear()
    x = torch.tensor([ids], device=model.device)
    out = model.model(input_ids=x, use_cache=False)
    last = out.last_hidden_state[0, start - 1 : -1]
    target = x[0, start:]
    lps, ents = [], []
    for i in range(0, last.shape[0], 256):
        logits = model.lm_head(last[i : i + 256]).float()
        lp = torch.log_softmax(logits, -1)
        lps.append(lp.gather(1, target[i : i + 256, None])[:, 0])
        ents.append(-(lp.exp() * lp).sum(-1))
    return torch.cat(lps).cpu(), torch.cat(ents).cpu()


def covering(offsets, c0, c1):
    """Token indices whose characters overlap [c0, c1)."""
    return [t for t, (a, b) in enumerate(offsets) if b > c0 and a < c1]


def item_positions(offsets, it):
    """Answer-relative token index sets of one citation: pre (the token whose
    state predicts the first ref token), ref, quote, item (ref to item end)."""
    if "ref" not in it or "quote" not in it:
        return None
    r0, r1 = it["ref"][0] + 1, it["ref"][1] - 1
    q0, q1 = it["quote"][0] + 1, it["quote"][1] - 1
    after_open = covering(offsets, r0, it["item_span"][1])
    first = after_open[0]
    # if the opening quote shares a token with the first ref character, that
    # token is the first one decided at `pre`
    if offsets[first][0] < r0 - 1:
        pass
    return {
        "pre": [first - 1],
        "ref": covering(offsets, r0, max(r1, r0 + 0)) or [first],
        "quote": covering(offsets, q0, q1) or covering(offsets, q0 - 1, q0),
        "item": covering(offsets, r0, it["item_span"][1]),
        # tokens that spell the ref and its closing quote
        "ref_closed": covering(offsets, r0, r1 + 1),
    }


def apply_edit(model, edit):
    """W <- W0 - alpha * u (u^T W0) on the listed columns of each layer's
    mlp.down_proj (float32 arithmetic from the bf16 base, cast back to bf16):
    the arithmetic of the weight files. Returns the originals for restore."""
    saved = {}
    for layer, spec in edit["layers"].items():
        w = layers_of(model)[int(layer)].mlp.down_proj.weight
        saved[int(layer)] = w.detach().clone()
        w32 = w.detach().float()
        u = torch.tensor(spec["u"], dtype=torch.float32, device=w.device)
        cols = torch.tensor(spec["columns"], device=w.device)
        sub = w32[:, cols]
        sub = sub - edit["alpha"] * spec.get("alpha_scale", 1.0) * torch.outer(u, u @ sub)
        w32[:, cols] = sub
        w.data.copy_(w32.to(w.dtype))
    return saved


def restore(model, saved):
    for layer, w0 in saved.items():
        layers_of(model)[layer].mlp.down_proj.weight.data.copy_(w0)


def read_jsonl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]
