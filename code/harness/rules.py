"""The edit rules, shared by the kernel's save_edit (inside the sandbox) and the host (before every 'edited' pass and
at the final submission). Astra's rules as stated in logs/exp8_math/runs/p50c/turns/t001/prompt.txt:
refused are the final norm, the input embedding / output layer, and any edit that only multiplies a whole tensor by a
constant. Also refused here: names outside the text model (vision tower, MTP head), a wrong shape, non-finite values.
Standard library + numpy only (torch tensors are converted by the caller)."""

import numpy as np

PREFIX = "model.language_model."
REFUSED = {
    PREFIX + "norm.weight": "the final norm",
    PREFIX + "embed_tokens.weight": "the input embedding / output layer (tied)",
    "lm_head.weight": "the output layer",
}
SCALE_TOL = 1e-6   # relative residual below which an edit counts as a whole-tensor scaling


class EditRefused(ValueError):
    pass


def text_names(weight_map):
    """the editable namespace: every text-model tensor (the refused ones stay listed, as Astra's NAMES did)."""
    return sorted(k for k in weight_map if k.startswith(PREFIX))


def check_edit(name, new, orig, names):
    """raise EditRefused if the edit breaks a rule; return a short note otherwise."""
    if name in REFUSED:
        raise EditRefused(f"save_edit refused: {name} is {REFUSED[name]} (out of scope)")
    if name not in names:
        raise EditRefused(f"save_edit refused: {name!r} is not a tensor of the text model (see NAMES)")
    new = np.asarray(new, dtype=np.float32)
    orig = np.asarray(orig, dtype=np.float32)
    if new.shape != orig.shape:
        raise EditRefused(f"save_edit refused: shape {new.shape} != {orig.shape} for {name}")
    if not np.isfinite(new).all():
        raise EditRefused(f"save_edit refused: non-finite values in {name}")
    o = orig.astype(np.float64).ravel()
    n = new.astype(np.float64).ravel()
    oo = float(o @ o)
    if oo > 0:
        c = float(n @ o) / oo
        resid = np.linalg.norm(n - c * o)
        if resid <= SCALE_TOL * max(np.linalg.norm(n), 1e-30) and abs(c - 1.0) > 1e-7:
            raise EditRefused(f"save_edit refused: the edit only multiplies {name} by a constant ({c:.6g})")
    elif not n.any():
        return "unchanged (zeros)"
    d = n - o
    rel = float(np.linalg.norm(d) / max(np.sqrt(oo), 1e-30))
    return f"relative change {rel:.4g}"
