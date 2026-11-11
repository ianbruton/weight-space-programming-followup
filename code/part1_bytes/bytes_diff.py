"""Compare base and edited Qwen3.5-0.8B safetensors on CPU; test W = W0 - k u (u^T W0).

Own script (nothing from the downloaded repositories). Usage:
  python -I bytes_diff.py BASE.safetensors EDITED.safetensors OUT_DIR
Writes OUT_DIR/bytes_result.json and OUT_DIR/u_layer{15,19}.npy.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def bits(t: torch.Tensor) -> np.ndarray:
    """Raw 16-bit patterns of a bf16 tensor (or bytes of anything else)."""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy()
    return t.contiguous().view(torch.uint8).numpy()


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(bits(t).tobytes()).hexdigest()


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def analyse_tensor(name: str, w0: torch.Tensor, w1: torch.Tensor) -> dict:
    b0, b1 = bits(w0), bits(w1)
    diff = b0 != b1
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    out = {
        "name": name,
        "shape": list(w0.shape),
        "changed_entries": int(diff.sum()),
        "changed_rows": int(len(rows)),
        "changed_columns": int(len(cols)),
        "columns": cols.tolist(),
        "addressed_entries": int(len(cols) * w0.shape[0]),
        "base_sha256": sha(w0),
        "edited_sha256": sha(w1),
    }
    # Work in float64 on the touched columns.
    W0 = w0.to(torch.float64)[:, cols]
    W1 = w1.to(torch.float64)[:, cols]
    D = W1 - W0  # (1024, 64): expected = -k u (u^T W0)
    U, S, Vt = torch.linalg.svd(D, full_matrices=False)
    out["delta_singular_values_top5"] = S[:5].tolist()
    out["delta_rank1_energy_share"] = float(S[0] ** 2 / (S**2).sum())
    u = U[:, 0]
    # Sign convention: make u^T W0 and the coefficient comparable.
    proj = u @ W0  # (64,) = u^T w0_j
    coef = u @ D  # (64,) = c_j, D_j ~ c_j u
    ratio = -coef / proj
    out["ratio_minus_c_over_proj"] = {
        "mean": float(ratio.mean()),
        "std": float(ratio.std()),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
    }
    k = float(ratio.mean())
    # Refine u: D = -k u p^T with p = u^T W0; alternate a few times (closed form,
    # not a fit of weights to a loss: rank-one factorisation of the observed delta).
    for _ in range(20):
        p = u @ W0
        u_new = -(D @ p) / (k * (p @ p))
        u = u_new / u_new.norm()
        k = float((-(u @ D) @ (u @ W0)) / ((u @ W0) @ (u @ W0)))
    out["k_refined"] = k
    # Reconstruct with k = 6 exactly and with refined k, round to bf16, count misses.
    for label, kk in (("k6", 6.0), ("k_refined", k)):
        for dt_label, dt in (("f32", torch.float32), ("f64", torch.float64)):
            uu = u.to(dt)
            W0d = w0.to(dt)[:, cols]
            Wr = W0d - kk * torch.outer(uu, uu @ W0d)
            rec = bits(bf16_round(Wr))
            tgt = bits(w1[:, cols])
            out[f"rebuild_{label}_{dt_label}_mismatched_entries"] = int(
                (rec != tgt).sum()
            )
    # Residual of the rank-one model, relative to delta norm.
    p = u @ W0
    model = -k * torch.outer(u, p)
    out["rank1_residual_rel"] = float((D - model).norm() / D.norm())
    # How many addressed entries round back to their original value.
    out["addressed_but_unchanged"] = out["addressed_entries"] - out["changed_entries"]
    # Columns: does the edit hit the columns that write u most? Rank of |u^T w0_j|.
    all_proj = (u @ w0.to(torch.float64)).abs()
    rank = torch.argsort(all_proj, descending=True)
    top64 = set(rank[:64].tolist())
    out["edited_cols_in_top64_by_abs_proj_on_u"] = len(top64 & set(cols.tolist()))
    return out, u.numpy().astype(np.float32)


def main() -> None:
    base_path, edit_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"base": base_path, "edited": edit_path}
    with safe_open(base_path, "pt") as fb, safe_open(edit_path, "pt") as fe:
        kb, ke = set(fb.keys()), set(fe.keys())
        result["n_tensors_base"] = len(kb)
        result["n_tensors_edited"] = len(ke)
        result["only_in_base"] = sorted(kb - ke)
        result["only_in_edited"] = sorted(ke - kb)
        n_params = 0
        n_lm = 0
        differing = []
        for name in sorted(kb & ke):
            a, b = fb.get_tensor(name), fe.get_tensor(name)
            n_params += a.numel()
            if name.startswith("model.language_model."):
                n_lm += a.numel()
            if a.shape != b.shape or a.dtype != b.dtype:
                differing.append(name)
                continue
            if not np.array_equal(bits(a), bits(b)):
                differing.append(name)
        result["n_params_release"] = n_params
        result["n_params_language_model"] = n_lm
        result["differing_tensors"] = differing
        us = {}
        result["tensors"] = []
        for name in differing:
            info, u = analyse_tensor(name, fb.get_tensor(name), fe.get_tensor(name))
            result["tensors"].append(info)
            layer = name.split(".layers.")[1].split(".")[0]
            us[layer] = u
            np.save(out_dir / f"u_layer{layer}.npy", u)
    if len(us) == 2:
        a, b = list(us.values())
        result["cos_u_between_layers"] = float(np.dot(a, b))
    result["changed_entries_total"] = sum(t["changed_entries"] for t in result["tensors"])
    (out_dir / "bytes_result.json").write_text(json.dumps(result, indent=1))
    slim = {k: v for k, v in result.items() if k != "tensors"}
    print(json.dumps(slim, indent=1))
    for t in result["tensors"]:
        print(json.dumps({k: v for k, v in t.items() if k != "columns"}, indent=1))


if __name__ == "__main__":
    main()
