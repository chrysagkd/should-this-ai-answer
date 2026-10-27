from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import yaml
from sksurv.datasets import load_aids

from pipeline import prepare_input

ROOT = Path(__file__).resolve().parent
GROUPS = ["clean", "missing_input", "unit_mismatch", "coding_mismatch", "endpoint_mismatch", "out_of_population"]


def summary(name, groups, withheld):
    df = pd.DataFrame({"group": groups, "withheld": withheld})
    pct = (100 * df.groupby("group").withheld.mean()).reindex(GROUPS).round(0).astype(int)
    invalid = df[df.group != "clean"]
    print(f"\n{name}: % withheld by the range check")
    print(pct.to_string())
    print(f"invalid requests answered: {int((~invalid.withheld).sum())} of {len(invalid)}; "
          f"well-formed withheld: {int(df[df.group == 'clean'].withheld.sum())} of {int((df.group == 'clean').sum())}")


#GBSG2
bundle = pickle.load(open(ROOT / "models" / "gbsg2_cox_v1.pkl", "rb"))
spec = yaml.safe_load(open(ROOT / "spec" / "gbsg2_cox_v1.yaml"))
ranges = spec["observed_ranges"]
requests = json.load(open(ROOT / "data" / "requests.json"))
groups, withheld = [], []
for r in requests:
    vec, _ = prepare_input(r, bundle)
    out = any(not (ranges[k]["p01"] <= x <= ranges[k]["p99"]) for k, x in zip(bundle["features"], vec))
    groups.append(r["group"]); withheld.append(out)
summary("GBSG2", groups, withheld)

#same preparation as 06
FEATURES = ["age", "cd4", "priorzdv", "karnof"]
KARNOF = ["70", "80", "90", "100"]
ECOG = ["0", "1", "2", "3", "4"]
X, _ = load_aids(endpoint="aids")
Xd = X[(X.age >= 30) & (X.age <= 55)][FEATURES].copy()
Xd["karnof"] = pd.Categorical(Xd["karnof"], categories=KARNOF, ordered=True).codes.astype(float)
Xd = Xd.astype(float)
med = Xd.median().to_dict()
lo, hi = Xd.quantile(0.01).to_dict(), Xd.quantile(0.99).to_dict()


def prepare(req):
    p = req["patient"]
    k = p["karnof"]
    if k is None:
        g = med["karnof"]
    elif k in KARNOF:
        g = float(KARNOF.index(k))
    elif k in ECOG:
        g = float(ECOG.index(k))
    else:
        g = med["karnof"]
    vals = [p["age"], p["cd4"], p["priorzdv"], g]
    return [med[c] if v is None else v for c, v in zip(FEATURES, vals)]


requests = json.load(open(ROOT / "data" / "requests_actg.json"))
groups = [r["group"] for r in requests]
withheld = [any(not (lo[c] <= x <= hi[c]) for c, x in zip(FEATURES, prepare(r))) for r in requests]
summary("ACTG", groups, withheld)
