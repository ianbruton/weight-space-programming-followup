"""Run once when the kernel starts (inside the sandbox). Defines what Astra's kernel had:
NAMES, load(name), load_original(name), save_edit(name, tensor), drop_edit(name), edited(), tok.
/model (read-only): the model's files; /practice (read-only); /work (read-write); /harness (read-only, this code)."""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/harness")
import rules as _rules  # noqa: E402

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
MODEL, PRACTICE, WORK = "/model", "/practice", "/work"
EDIT_DIR = WORK + "/edited"
os.makedirs(EDIT_DIR, exist_ok=True)
_WMAP = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]
NAMES = _rules.text_names(_WMAP)
_NAMESET = set(NAMES)


def load_original(name):
    """the original tensor as float32 torch tensor."""
    if name not in _WMAP:
        raise KeyError(f"unknown tensor {name!r}; see NAMES")
    with safe_open(MODEL + "/" + _WMAP[name], framework="pt") as f:
        return f.get_tensor(name).float()


def load(name):
    """your current edit of the tensor if there is one, else the original (float32 torch tensor)."""
    p = EDIT_DIR + "/" + name + ".npy"
    if os.path.exists(p):
        return torch.from_numpy(np.load(p).astype(np.float32))
    return load_original(name)


def save_edit(name, tensor):
    """save an edit (refused: final norm, embedding/output layer, whole-tensor scalings, other names)."""
    a = tensor.detach().float().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor, np.float32)
    note = _rules.check_edit(name, a, load_original(name).numpy(), _NAMESET)
    np.save(EDIT_DIR + "/" + name + ".npy", a.astype(np.float32))
    return f"saved {name}: {note}"


def drop_edit(name):
    p = EDIT_DIR + "/" + name + ".npy"
    if os.path.exists(p):
        os.remove(p)
        return f"dropped {name}"
    return f"no edit of {name}"


def edited():
    return sorted(f[:-4] for f in os.listdir(EDIT_DIR) if f.endswith(".npy"))


try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
except Exception as _e:  # noqa: BLE001
    tok = None
    print("tokenizer not loaded:", _e)
