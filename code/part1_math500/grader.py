"""Exact MATH answer grader for the scale study (own code; standard library + sympy only).

Extraction follows the authors' rule as read in logs/exp8_math/code/mathqa.py (not imported, not run):
  - the text read is what follows the last '</think>' if there is one, else the whole text;
  - the answer is the content of the LAST '\\boxed{..}' (or '\\fbox{..}'), brace matched; '\\boxed 5' is read up to
    the next '$' or line end; an unclosed or empty last box is no answer.
`extract_strict` is the stricter variant: a thinking-mode text without '</think>' has no answer.

Equivalence (`equiv`) is mine (math-verify is not on this machine): string equality after a MATH-style
normalisation, then element-wise for tuples / lists / intervals, then a numeric check through sympy on a small
LaTeX subset (fractions, roots, powers, pi, products), then a symbolic check (sympy expand of the difference) for
expressions in the same single-letter variables. A trailing \\text{..} holding only unit words is dropped (added
2026-11-10 after the 4B practice pass: 20\\text{ miles} against 20). Anything it cannot parse is compared as a string.
"""

import re
from fractions import Fraction

BOX = re.compile(r"\\boxed|\\fbox")


# ------------------------------------------------------------------ extraction
def answer_text(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    """the authors' rule: last box of the text after the last </think> (or of the whole text)."""
    return last_boxed(answer_text(text))


def extract_strict(text, thinking):
    """a thinking text that never closed its thinking has no answer."""
    if thinking and "</think>" not in text:
        return None
    return extract(text)


# ------------------------------------------------------------------ normalisation
def _unwrap(s, cmd):
    """replace \\cmd{X} by X (brace matched), every occurrence."""
    out, i, key = [], 0, "\\" + cmd + "{"
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len(key) - 1, 0
        for m in range(k, len(s)):
            if s[m] == "{":
                depth += 1
            elif s[m] == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[k + 1:m])
                    i = m + 1
                    break
        else:
            out.append(s[j:])
            return "".join(out)


def _fix_fracs(s):
    # \frac12 -> \frac{1}{2}; \frac1{2} -> \frac{1}{2}
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    return s


UNIT_WORDS = (r"(?:square|sq\.?|cubic|miles?|mi|km|kilometers?|meters?|metres?|m|cm|mm|feet|foot|ft|inches|inch|in|"
              r"yards?|units?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|dollars?|cents?|"
              r"pounds?|lbs?|ounces?|oz|kg|grams?|g|liters?|litres?|gallons?|mph|degrees?|people|students?|apples?)")
UNIT_TEXT = re.compile(r"\\(?:text|mbox|mathrm)\{\s*" + UNIT_WORDS + r"(?:\s+" + UNIT_WORDS + r")*\s*(?:\^\{?[23]\}?)?\s*\}"
                       r"(?:\^\{?[23]\}?)?\s*$")


def norm(s):
    if s is None:
        return None
    s = s.strip()
    # a trailing \text{ <unit words> } after an answer (20\text{ miles}, 32\text{ square cm}); only inside \text,
    # so a bare variable (5x) is never dropped
    t = UNIT_TEXT.sub("", s).strip()
    if t:
        s = t
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""), ("~", ""),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\circ", ""), ("\\%", ""), ("%", ""), ("\\$", ""),
                 ("$", ""), ("\\displaystyle", ""), ("\n", ""), ("\\le", "\\le"), ("\\dots", "..."),
                 ("\\ldots", "...")):
        s = s.replace(a, b)
    for cmd in ("text", "textbf", "mathrm", "mathbf", "boxed", "mbox", "operatorname"):
        s = _unwrap(s, cmd)
    s = s.replace(" ", "")
    # degrees / units words commonly appended
    s = re.sub(r"(degrees?|units?|cm|inches|inch|feet|meters|dollars|cents|sq)(\^[23])?$", "", s)
    s = re.sub(r"^[a-z]\\in", "", s)                       # x \in [-2,7]
    if re.fullmatch(r"\([A-Ea-e]\)", s):                    # (E) -> E
        s = s[1]
    if s.startswith("\\{") and s.endswith("\\}") and "," not in s:   # \{-4\} -> -4
        s = s[2:-2]
    # matrices: a/b entries -> \frac{a}{b}
    if "pmatrix" in s or "bmatrix" in s:
        s = re.sub(r"(-?\d+)/(\d+)", r"\\frac{\1}{\2}", s)
    # a leading 'x=' or 'k=' (short left side, single '=')
    if s.count("=") == 1 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = s.rstrip(".")
    s = _fix_fracs(s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    # base subscripts: 4210_{5} -> 4210_5
    s = re.sub(r"_\{(\w+)\}", r"_\1", s)
    # thousands separators in plain integers: 1,000 / 1{,}000
    s = s.replace("{,}", ",")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if s.startswith("."):
        s = "0" + s
    return s


# ------------------------------------------------------------------ numeric check (sympy, small LaTeX subset)
_SYMPY = []


def _sp():
    if not _SYMPY:
        import sympy  # noqa: PLC0415
        _SYMPY.append(sympy)
    return _SYMPY[0]


def _latex_to_py(s):
    """a small LaTeX subset to a sympy-parsable string, or None."""
    t = s
    for _ in range(6):   # nested \frac / \sqrt
        t2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
        t2 = re.sub(r"\\sqrt\[(\d+)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", t2)
        t2 = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t2)
        if t2 == t:
            break
        t = t2
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    t = t.replace("{", "(").replace("}", ")")
    if "\\" in t or re.search(r"[A-Za-z]{2,}", re.sub(r"(sqrt|pi)", "", t)):
        return None
    if re.search(r"[a-oq-zA-Z]", re.sub(r"(sqrt|pi)", "", t)):   # free variables: leave to string compare
        return None
    # implicit products: 2pi, 3sqrt(2), )( , 2(
    t = re.sub(r"(\d)\s*(pi|sqrt|\()", r"\1*\2", t)
    t = re.sub(r"\)\s*(\(|\d|pi|sqrt)", r")*\1", t)
    t = re.sub(r"pi\s*(\(|\d|sqrt)", r"pi*\1", t)
    return t


def _num(s):
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    py = _latex_to_py(s)
    if py is None:
        return None
    try:
        sp = _sp()
        v = sp.sympify(py, evaluate=True)
        if v.free_symbols:
            return None
        return v
    except Exception:  # noqa: BLE001
        return None


def _num_equal(a, b):
    x, y = _num(a), _num(b)
    if x is None or y is None:
        return False
    if isinstance(x, Fraction) and isinstance(y, Fraction):
        return x == y
    try:
        sp = _sp()
        xv, yv = complex(sp.N(sp.sympify(x), 30)), complex(sp.N(sp.sympify(y), 30))
        return abs(xv - yv) < 1e-9 * max(1.0, abs(yv))
    except Exception:  # noqa: BLE001
        return False


def _split_top(s):
    """split on top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _atom_equal(a, b):
    return a == b or _num_equal(a, b)


def _drop_base(s):
    """52_8 -> 52 (a number in a stated base); math-verify reads the two as equal, so this does too."""
    m = re.fullmatch(r"(\w+)_\w+", s)
    return m.group(1) if m else s


def equiv(pred, gold):
    if pred is None or gold is None:
        return False
    a, b = norm(pred), norm(gold)
    if a == b or _drop_base(a) == _drop_base(b):
        return True
    if _sym_equal(a, b):        # before the tuple rule, which would split (x+2)(x+1) as a pair
        return True
    # tuples / intervals: same brackets, element-wise
    if len(a) > 1 and len(b) > 1 and a[0] in "([" and b[0] in "([" and a[-1] in ")]" and b[-1] in ")]":
        if (a[0], a[-1]) != (b[0], b[-1]):
            return False
        pa, pb = _split_top(a[1:-1]), _split_top(b[1:-1])
        return len(pa) == len(pb) and all(_atom_equal(x, y) for x, y in zip(pa, pb))
    # lists of solutions ('1,2' or '\{1,2\}'): unordered
    sa, sb = a.replace("\\{", "").replace("\\}", ""), b.replace("\\{", "").replace("\\}", "")
    pa, pb = _split_top(sa), _split_top(sb)
    if len(pb) > 1 or len(pa) > 1:
        if len(pa) != len(pb):
            return False
        used = [False] * len(pb)
        for x in pa:
            for j, y in enumerate(pb):
                if not used[j] and _atom_equal(x, y):
                    used[j] = True
                    break
            else:
                return False
        return True
    # text answers: case-insensitive
    if a.lower() == b.lower():
        return True
    return _num_equal(a, b) or _sym_equal(a, b)


def _sym_py(s):
    """like _latex_to_py, but single-letter variables allowed (implicit products made explicit), or None."""
    t = s
    for _ in range(6):
        t2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
        t2 = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t2)
        if t2 == t:
            break
        t = t2
    t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
    t = t.replace("{", "(").replace("}", ")")
    if "\\" in t or "=" in t or "," in t or re.search(r"[A-Za-z]{2,}", re.sub(r"(sqrt|pi)", "", t)):
        return None
    t = re.sub(r"(\d|\))\s*([a-zA-Z(])", r"\1*\2", t)          # 2x, 2(, )(, )x
    t = re.sub(r"(?<![a-z])([a-zA-Z])\s*(\(|\d)", r"\1*\2", t)  # x(, x2  (not sqrt( / pi)
    t = t.replace("sqrt*(", "sqrt(").replace("pi*(", "pi*(")
    return t


def _sym_equal(a, b):
    """symbolic equality of two expressions in the same single-letter variables ((x+2)(x+1) = (x+1)(x+2))."""
    pa, pb = _sym_py(a), _sym_py(b)
    if pa is None or pb is None:
        return False
    try:
        sp = _sp()
        x, y = sp.sympify(pa), sp.sympify(pb)
        if not x.free_symbols or x.free_symbols != y.free_symbols:
            return False
        return sp.expand(x - y) == 0
    except Exception:  # noqa: BLE001
        return False


def grade(text, gold, thinking):
    """{pred, correct (authors' extraction), pred_strict, correct_strict}."""
    p = extract(text)
    ps = extract_strict(text, thinking)
    return {"pred": p if p is None or len(p) <= 200 else p[:200], "correct": equiv(p, gold),
            "correct_strict": equiv(ps, gold) if ps is not None else False}


# ------------------------------------------------------------------ looping (own rule, from the authors' description)
def verbatim_loop_end(text, max_period=200, min_repeats=3, min_span=40):
    """the text (without its last, possibly cut, word) ends with a block of 1..200 words repeated in a row at least
    3 times over at least 40 words."""
    w = text.split()[:-1]
    n = len(w)
    for p in range(1, max_period + 1):
        reps = max(min_repeats, -(-min_span // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            return True
    return False


def repetitive_end(text, window=200, k=4, min_count=3):
    """in the last 200 words, the share of 4-word positions whose phrase occurs 3+ times there is at least half."""
    w = text.split()[-window:]
    grams = [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]
    if not grams:
        return False
    cnt = {}
    for g in grams:
        cnt[g] = cnt.get(g, 0) + 1
    return sum(cnt[g] >= min_count for g in grams) >= len(grams) / 2
