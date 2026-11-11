"""CPU, read only: finding citations held, pooled per arm, with a bootstrap
over findings for the difference between arms (the citations of one finding
are not independent, so findings are resampled, the same draw for both arms;
every run of an arm is pooled). Printed for all findings, the practice
findings and the held-out findings of this study's split.

Usage: /usr/bin/python3 -B -I n_boot.py ARM=RUN_FOLDER ... -- ARM_A:ARM_B ...
Several folders may share an ARM. Seed fixed (20261110), 10,000 draws.
"""

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

args = sys.argv[1:]
cut = args.index("--")
runs, pairs = args[:cut], [p.split(":") for p in args[cut + 1:]]


def split_of(key):
    return "heldout" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 < 2 else "practice"


arm = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # arm -> key -> [ok, cited]
nruns = defaultdict(int)
for r in runs:
    name, folder = r.split("=", 1)
    nruns[name] += 1
    for line in (Path(folder) / "proposals.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["key"].startswith("probe:"):
            continue
        c = (rec.get("check") or {}).get("citations") or {"cited": 0, "ok": 0}
        arm[name][rec["key"]][0] += c["ok"]
        arm[name][rec["key"]][1] += c["cited"]
        arm[name][rec["key"]]  # a finding with no citation still counts as a finding

keys_all = sorted(set().union(*[set(v) for v in arm.values()]))
subsets = {"all": keys_all,
           "practice": [k for k in keys_all if split_of(k) == "practice"],
           "heldout": [k for k in keys_all if split_of(k) == "heldout"]}


def share(name, keys):
    ok = sum(arm[name][k][0] for k in keys if k in arm[name])
    cited = sum(arm[name][k][1] for k in keys if k in arm[name])
    return ok, cited


out = {}
for sub, keys in subsets.items():
    print(f"== {sub} ({len(keys)} findings)")
    if not keys:
        continue
    for name in arm:
        ok, cited = share(name, keys)
        print(f"   {name:12s} {nruns[name]} run(s)  {ok}/{cited} ({100 * ok / cited:.1f}%)")
        out.setdefault(sub, {})[name] = [ok, cited]
    for a, b in pairs:
        rng = random.Random(20261110)
        oa, ca = share(a, keys)
        ob, cb = share(b, keys)
        point = oa / ca - ob / cb
        draws = []
        for _ in range(10000):
            ks = [keys[rng.randrange(len(keys))] for _ in keys]
            xa, ya = share(a, ks)
            xb, yb = share(b, ks)
            if ya and yb:
                draws.append(xa / ya - xb / yb)
        draws.sort()
        lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
        above = sum(1 for d in draws if d > 0) / len(draws)
        print(f"   {a} - {b}: {100 * point:+.1f} points, 95% interval {100 * lo:+.1f} to {100 * hi:+.1f}, "
              f"draws above zero {100 * above:.1f}%")
        out[sub][f"{a}-{b}"] = [point, lo, hi, above]
