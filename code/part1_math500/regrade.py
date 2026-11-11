"""Re-grade released MATH-500 samples (no generation): own grader, own token counts.

Usage: python -I regrade.py CARD_EVAL_DIR LOGS_EXP8_DIR TOKENIZER_DIR OUT_JSON
Inputs (read only):
  CARD_EVAL_DIR = the edited model repository's eval/ folder:
      samples_{base,edited}_math500.jsonl + scores_*.json          (the model card's re-run)
  LOGS_EXP8_DIR = exp8_math/ of the authors' logs repository:
      eval/{unedited,p50c}/parts/math500_s*/                        (the authors' runs)
"""

import gzip
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

CAP = 32768


# ---------- answer extraction ----------
def last_boxed(text: str) -> str | None:
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = idx
    while i < len(text) and text[i] != "{":
        if text[i] == " " and text.startswith("\\boxed ", idx):
            # \boxed 5 form
            m = re.match(r"\\boxed (\S+)", text[idx:])
            return m.group(1) if m else None
        i += 1
    if i >= len(text):
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None  # unclosed box


# ---------- normalisation ----------
def _fix_fracs(s: str) -> str:
    # \frac12 -> \frac{1}{2}, \frac1{2} -> \frac{1}{2}
    out = ""
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{"):
            out += p
            continue
        if len(p) < 2:
            return s
        a, rest = p[0], p[1:]
        if rest.startswith("{"):
            out += "{" + a + "}" + rest
        else:
            out += "{" + a + "}{" + rest[0] + "}" + rest[1:]
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _strip_text_units(s: str) -> str:
    s = re.sub(r"\\(?:text|textbf|mbox|mathrm)\{\s*([^{}]*)\}", r"\1", s)
    return s


def normalize(ans: str) -> str:
    s = ans.strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = _strip_text_units(s)
    s = s.replace("\\displaystyle", "")
    # drop trailing unit words commonly appended (e.g. "5 cm", "12 inches")
    s = re.sub(
        r"(?<=[\d}])\s*(?:units?|cm|m|inches|inch|feet|ft|degrees|dollars|cents|"
        r"square units|sq units|meters|miles|hours|minutes|seconds|days)\s*$",
        "",
        s,
    )
    s = s.replace(" ", "")
    s = s.rstrip(".")
    # "x=5" -> "5" when a single short lhs
    m = re.fullmatch(r"[a-zA-Z]\w?=(.+)", s)
    if m and "," not in s:
        s = m.group(1)
    s = _fix_sqrt(_fix_fracs(s))
    # leading zero for decimals
    s = re.sub(r"(?<![\d])\.(\d)", r"0.\1", s)
    # thousands separators "1,000" -> "1000" (only pure digit groups)
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    s = s.replace("{,}", "")
    return s


def to_number(s: str):
    """Numeric value of simple forms: ints, decimals, \\frac{a}{b}, -\\frac{a}{b}, a/b."""
    t = s
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", t)
    if m:
        v = Fraction(int(m.group(2)), int(m.group(3)))
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", t)
    if m and int(m.group(2)) != 0:
        return Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def equivalent(pred: str, gold: str) -> bool:
    a, b = normalize(pred), normalize(gold)
    if a == b:
        return True
    na, nb = to_number(a), to_number(b)
    if na is not None and nb is not None:
        return abs(float(na) - float(nb)) <= 1e-6 * max(1.0, abs(float(nb)))
    # unordered comma lists of the same items (e.g. roots)
    if "," in a and "," in b:
        sa = sorted(x.strip("()[]") for x in a.split(","))
        sb = sorted(x.strip("()[]") for x in b.split(","))
        if sa == sb and a[:1] == b[:1]:
            return True
    return False


def grade(text: str, gold: str) -> str:
    # Only the answer after the thinking counts if a </think> is present.
    pred = last_boxed(text.split("</think>")[-1]) if "</think>" in text else last_boxed(text)
    if pred is None:
        return "NO_BOX"
    return "CORRECT" if equivalent(pred, gold) else "WRONG"


# ---------- runs ----------
def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def tally(rows, tok, flags_first):
    out = {"n": len(rows), "CORRECT": 0, "WRONG": 0, "NO_BOX": 0, "at_cap": 0,
           "tokens": [], "agree_with_authors_flag": 0, "disagreements": []}
    for r in rows:
        g = grade(r["text"], r["answer"])
        out[g] += 1
        n_tok = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        out["tokens"].append(n_tok)
        if n_tok >= CAP - 64:
            out["at_cap"] += 1
        fl = flags_first.get(r["id"])
        if fl is not None:
            mine = 1 if g == "CORRECT" else 0
            if mine == fl:
                out["agree_with_authors_flag"] += 1
            else:
                out["disagreements"].append(
                    {"id": r["id"], "gold": r["answer"], "mine": g, "authors": fl,
                     "pred": last_boxed(r["text"])}
                )
    t = sorted(out["tokens"])
    out["tokens_median"] = t[len(t) // 2] if t else None
    out["tokens_max"] = t[-1] if t else None
    out["accuracy"] = out["CORRECT"] / out["n"] if out["n"] else None
    out["at_cap_share"] = out["at_cap"] / out["n"] if out["n"] else None
    del out["tokens"]
    return out


def flags_from_scores(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {p["id"]: p["correct"][0] for p in d["per_problem"]}


def run_logs(run_dir: Path, tok):
    rows, flags, all_flags, cut, total = [], {}, [], 0, 0
    pred_first_boxed = 0
    for part in sorted(run_dir.glob("parts/math500_s*")):
        sc = json.loads((part / "scores.json").read_text())["math500"]
        cut += sc["cut"]
        total += sc["total"]
        pp = json.loads((part / "math500_per_problem.json").read_text())
        for p in pp:
            flags[p["id"]] = p["correct"][0]
            all_flags += p["correct"]
            pred_first_boxed += p.get("pred_first") is not None
        with gzip.open(part / "math500_first_samples.jsonl.gz", "rt") as f:
            rows += [json.loads(line) for line in f]
    res = tally(rows, tok, flags)
    res["authors_flags_mean_all_samples"] = sum(all_flags) / len(all_flags)
    res["authors_flags_n"] = len(all_flags)
    res["authors_cut_share_all_samples"] = cut / total
    res["authors_cut_total"] = [cut, total]
    res["first_samples_with_pred"] = pred_first_boxed
    res["problems"] = len(flags)
    res["authors_first_sample_acc_over_500"] = sum(flags.values()) / len(flags)
    # same subset as the texts: the authors' first-sample flags on those problems
    ids = [r["id"] for r in rows]
    res["authors_first_sample_acc_on_text_subset"] = sum(flags[i] for i in ids) / len(ids)
    return res, ids


def main():
    card, exp8, tokdir, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    tok = load_tokenizer(tokdir)
    result = {}
    # 1. model card's samples
    for which in ("base", "edited"):
        rows = [json.loads(line) for line in
                open(card / f"samples_{which}_math500.jsonl")]
        scores = card / f"scores_{which}_math500.json"
        flags = flags_from_scores(scores)
        summ = json.loads(scores.read_text())["summary"]
        res = tally(rows, tok, flags)
        res["card_summary"] = summ
        res["card_flags_first_sample_on_text_subset"] = (
            sum(flags[r["id"]] for r in rows) / len(rows))
        result[f"card_{which}"] = res
    # 2. authors' runs
    ids_by_run = {}
    for run in ("unedited", "p50c"):
        res, ids = run_logs(exp8 / "eval" / run, tok)
        result[f"authors_{run}"] = res
        ids_by_run[run] = ids
    result["same_text_subset_unedited_vs_p50c"] = ids_by_run["unedited"] == ids_by_run["p50c"]
    card_ids_base = [json.loads(l)["id"] for l in open(card / "samples_base_math500.jsonl")]
    result["card_sample_ids_within_authors_text_subset"] = sum(
        i in set(ids_by_run["unedited"]) for i in card_ids_base)
    out.write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        if isinstance(v, dict):
            print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "disagreements"}))
            for d in v["disagreements"][:12]:
                print("   DISAGREE", json.dumps(d)[:300])
        else:
            print(k, v)


if __name__ == "__main__":
    main()
