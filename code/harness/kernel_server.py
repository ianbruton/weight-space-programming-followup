"""The persistent kernel (runs inside the sandbox): executes /kernel/inbox/<seq>.py in one namespace that survives
between runs, writes /kernel/outbox/<seq>.out then /kernel/outbox/<seq>.done. Like a notebook: if the last statement
is an expression, its value is shown. The time limit is enforced from outside (the host kills and restarts the kernel)."""
import ast
import contextlib
import io
import os
import time
import traceback

K = "/kernel"
INBOX, OUTBOX = K + "/inbox", K + "/outbox"
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)
NS = {"__name__": "__main__"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        exec(compile(open("/harness/prelude.py").read(), "prelude", "exec"), NS)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
open(K + "/ready", "w").write(buf.getvalue())


def run(code):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            tree = ast.parse(code, "<cell>")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "<cell>", "exec"), NS)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), NS)
                if v is not None:
                    print(repr(v))
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
    return out.getvalue()


while True:
    todo = sorted(f for f in os.listdir(INBOX) if f.endswith(".py"))
    if not todo:
        time.sleep(0.2)
        continue
    f = todo[0]
    seq = f[:-3]
    code = open(INBOX + "/" + f).read()
    os.remove(INBOX + "/" + f)
    t0 = time.time()
    res = run(code)
    open(OUTBOX + "/" + seq + ".out", "w").write(res)
    open(OUTBOX + "/" + seq + ".done", "w").write(str(round(time.time() - t0, 2)))
