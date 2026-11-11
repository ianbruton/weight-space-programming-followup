"""Search for a direction u (per layer) that rebuilds every edited bf16 entry exactly.

Model: W = bf16( W0 - 6 u (u^T W0) ) on the 64 edited columns. Each edited bf16 value
fixes an interval of real values that round to it; for fixed p = u^T W0 every row i
gives an interval for a_i = 6 u_i. Alternate: p from u, a from interval midpoints.
Own script. Usage: python -I u_exact.py BASE EDITED OUT_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

K = 6.0


def bf16_bounds(w1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real interval [lo, hi] of float64 values that round (RNE) to each bf16 value."""
    v = w1.to(torch.float64)
    b = w1.contiguous().view(torch.int16).to(torch.int32)
    up = (b + 1).to(torch.int16).view(torch.bfloat16).to(torch.float64)
    dn = (b - 1).to(torch.int16).view(torch.bfloat16).to(torch.float64)
    # For negative numbers bit+1 moves away from zero; order the neighbours.
    nb_hi = torch.maximum(up, dn)
    nb_lo = torch.minimum(up, dn)
    return (v + nb_lo) / 2, (v + nb_hi) / 2


def rebuild_misses(w0c, w1c, u, k, dtype):
    W0 = w0c.to(dtype)
    uu = torch.as_tensor(u, dtype=dtype)
    r = (W0 - k * torch.outer(uu, uu @ W0)).to(torch.bfloat16)
    a = r.contiguous().view(torch.int16)
    b = w1c.contiguous().view(torch.int16)
    miss = a != b
    ulps = (a.to(torch.int32) - b.to(torch.int32)).abs()[miss]
    return int(miss.sum()), (int(ulps.max()) if miss.any() else 0)


def solve(w0: torch.Tensor, w1: torch.Tensor, u0: np.ndarray, iters: int = 60):
    diff = (w0.view(torch.int16) != w1.view(torch.int16)).any(dim=0)
    cols = torch.where(diff)[0]
    w0c, w1c = w0[:, cols], w1[:, cols]
    W0 = w0c.to(torch.float64)
    lo, hi = bf16_bounds(w1c)
    u = torch.as_tensor(u0, dtype=torch.float64)
    u = u / u.norm()
    best = None
    for it in range(iters):
        p = u @ W0  # (64,)
        # W0_ij - a_i p_j in [lo_ij, hi_ij]  ->  a_i in [(W0-hi)/p, (W0-lo)/p] (p>0)
        x1 = (W0 - hi) / p
        x2 = (W0 - lo) / p
        amin = torch.minimum(x1, x2).max(dim=1).values
        amax = torch.maximum(x1, x2).min(dim=1).values
        empty = int((amin > amax).sum())
        a = (amin + amax) / 2
        u_new = a / K
        # keep the scale consistent with K: u must be unit; renormalise and let the
        # next p absorb it.
        u = u_new / u_new.norm()
        m32 = rebuild_misses(w0c, w1c, u.numpy(), K, torch.float32)
        rec = (it, empty, float(u_new.norm()), m32)
        if best is None or m32[0] < best[3][0]:
            best = (it, empty, float(u_new.norm()), m32, u.clone())
    return cols, best


def main():
    base, edited, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    res = {}
    with safe_open(base, "pt") as fb, safe_open(edited, "pt") as fe:
        for layer in (15, 19):
            name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            u0 = np.load(out / f"u_layer{layer}.npy")
            cols, best = solve(fb.get_tensor(name), fe.get_tensor(name), u0)
            it, empty, norm_a_over_k, (miss, maxulp), u = best
            np.save(out / f"u_layer{layer}_refined.npy", u.numpy())
            w0 = fb.get_tensor(name)[:, cols]
            w1 = fe.get_tensor(name)[:, cols]
            m64 = rebuild_misses(w0, w1, u.numpy(), K, torch.float64)
            res[layer] = {
                "best_iter": it,
                "rows_with_empty_interval": empty,
                "norm_of_a_over_6": norm_a_over_k,
                "rebuild_k6_f32_misses_of_65536": miss,
                "rebuild_k6_f32_max_ulp": maxulp,
                "rebuild_k6_f64_misses_of_65536": m64[0],
                "rebuild_k6_f64_max_ulp": m64[1],
                "cos_with_svd_u": float(np.dot(u.numpy(), u0 / np.linalg.norm(u0))),
            }
    a = np.load(out / "u_layer15_refined.npy")
    b = np.load(out / "u_layer19_refined.npy")
    res["cos_u15_u19"] = float(np.dot(a, b))
    (out / "u_exact_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
