from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import mahalanobis
from sksurv.datasets import load_aids
from sksurv.linear_model import CoxPHSurvivalAnalysis

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260903)
N, N_BOOT = 100, 200

AGE_MIN, AGE_MAX = 30, 55
FEATURES = ["age", "cd4", "priorzdv", "karnof"]
KARNOF = ["70", "80", "90", "100"]

# Karnofsky -> ECOG standard published crosswalk
KARNOF_TO_ECOG = {"100": "0", "90": "0", "80": "1", "70": "1"}
ECOG = ["0", "1", "2", "3", "4"]


def encode(df):
    out = df[FEATURES].copy()
    out["karnof"] = pd.Categorical(
        out["karnof"], categories=KARNOF, ordered=True).codes.astype(float)
    return out.astype(float)


#fit both
X, y_aids = load_aids(endpoint="aids")
_, y_death = load_aids(endpoint="death")

in_pop = (X.age >= AGE_MIN) & (X.age <= AGE_MAX)
X_dev, X_out = X[in_pop], X[~in_pop]
Xd = encode(X_dev)

m_aids = CoxPHSurvivalAnalysis().fit(Xd, y_aids[in_pop.values])
m_death = CoxPHSurvivalAnalysis().fit(Xd, y_death[in_pop.values])

print(f"development set: {len(X_dev)}   held out: {len(X_out)}")
print(f"modelled endpoint : progression to AIDS "
      f"({y_aids['censor'][in_pop.values].sum()} events)")
print(f"other endpoint    : death "
      f"({y_death['censor_d'][in_pop.values].sum()} events)")

risk = m_aids.predict(Xd)
edges = np.quantile(risk, np.linspace(0, 1, 11))[1:-1]
edges_d = np.quantile(m_death.predict(Xd), np.linspace(0, 1, 11))[1:-1]

#endpoint disagreement
# Same patients, same covariates, two endpoints. How often does the risk decile differ? This is what a reader loses by misreading it.
dec_a = np.searchsorted(edges, risk) + 1
dec_d = np.searchsorted(edges_d, m_death.predict(Xd)) + 1
diff = np.abs(dec_a - dec_d)
print(f"\nendpoint disagreement on the same patients:")
print(f"  decile differs      : {100*(diff>0).mean():.0f}%")
print(f"  differs by 2 or more: {100*(diff>=2).mean():.0f}%")
print(f"  median |difference| : {np.median(diff):.1f}")

# spec
spec = {
    "model_id": "actg_cox_v1",
    "substrate": "ACTG clinical trial cohort",
    "endpoint": {"modelled": "progression_to_aids",
                 "not_valid_for": ["death", "overall_survival"]},
    "population": {"age": {"min": float(AGE_MIN), "max": float(AGE_MAX)}},
    "required_inputs": FEATURES,
    "input_units": {"priorzdv": "months", "cd4": "cells_per_mm3",
                    "age": "years"},
    "coding": {"karnof": {"scheme": "karnofsky", "allowed": KARNOF}},
}
with open(ROOT / "spec" / "actg_cox_v1.yaml", "w") as f:
    yaml.safe_dump(spec, f, sort_keys=False)

# requests
def base(pid, r, group):
    return {"request_id": f"{group}_{pid}", "group": group,
            "patient": {"age": float(r.age), "cd4": float(r.cd4),
                        "priorzdv": float(r.priorzdv),
                        "karnof": str(r.karnof)},
            "requested_endpoint": "progression_to_aids",
            "units": {"priorzdv": "months"},
            "coding": {"karnof": "karnofsky"}}


reqs = []
for pid in RNG.choice(X_dev.index, N, replace=False):
    reqs.append(base(pid, X_dev.loc[pid], "clean"))
for pid in RNG.choice(X_dev.index, N, replace=False):
    r = base(pid, X_dev.loc[pid], "missing_input")
    r["patient"]["cd4"] = None
    reqs.append(r)
for pid in RNG.choice(X_dev.index, N, replace=False):
    r = base(pid, X_dev.loc[pid], "unit_mismatch")
    r["patient"]["priorzdv"] /= 12.0          # months -> years
    r["units"]["priorzdv"] = "years"
    reqs.append(r)
for pid in RNG.choice(X_dev.index, N, replace=False):
    r = base(pid, X_dev.loc[pid], "coding_mismatch")
    r["patient"]["karnof"] = KARNOF_TO_ECOG[r["patient"]["karnof"]]
    r["coding"]["karnof"] = "ecog"
    reqs.append(r)
for pid in RNG.choice(X_dev.index, N, replace=False):
    r = base(pid, X_dev.loc[pid], "endpoint_mismatch")
    r["requested_endpoint"] = "death"
    reqs.append(r)
for pid in RNG.choice(X_out.index, min(N, len(X_out)), replace=False):
    reqs.append(base(pid, X_out.loc[pid], "out_of_population"))

json.dump(reqs, open(ROOT / "data" / "requests_actg.json", "w"), indent=1)

# pipeline
med = Xd.median().to_dict()


def prepare(req):
    p = req["patient"]
    k = p["karnof"]
    if k is None:
        g = med["karnof"]
    elif k in KARNOF:
        g = float(KARNOF.index(k))
    elif k in ECOG:
        g = float(ECOG.index(k))       # read ordinally in its own scheme
    else:
        g = med["karnof"]
    vals = [p["age"], p["cd4"], p["priorzdv"], g]
    vals = [med[c] if v is None else v for c, v in zip(FEATURES, vals)]
    return np.array(vals, dtype=float)


# mechanism A
boot = []
Xv, yv = Xd.values, y_aids[in_pop.values]
for _ in range(N_BOOT):
    i = RNG.choice(len(Xv), len(Xv), replace=True)
    try:
        boot.append(CoxPHSurvivalAnalysis().fit(Xv[i], yv[i]).coef_)
    except Exception:
        pass
boot = np.array(boot)
mu, vi = Xd.mean().values, np.linalg.inv(np.cov(Xv, rowvar=False))


def scores(v):
    return (v @ boot.T).std(), mahalanobis(v, mu, vi)


cl = [scores(prepare(r)) for r in reqs if r["group"] == "clean"]
d95 = np.percentile([c[0] for c in cl], 95)
m95 = np.percentile([c[1] for c in cl], 95)
thr = np.percentile([max(a / d95, b / m95) for a, b in cl], 95)


def mech_a(r):
    d, m = scores(prepare(r))
    return max(d / d95, m / m95) <= thr


#mechanism B
def mech_b(r):
    p = r["patient"]
    for k in spec["required_inputs"]:
        if p.get(k) is None:
            return False
    for k, u in spec["input_units"].items():
        if k in r["units"] and r["units"][k] != u:
            return False
    for k, c in spec["coding"].items():
        if k in r["coding"] and r["coding"][k] != c["scheme"]:
            return False
    if r["requested_endpoint"] != spec["endpoint"]["modelled"]:
        return False
    for k, rg in spec["population"].items():
        if not (rg["min"] <= p[k] <= rg["max"]):
            return False
    return True


#run
rows = []
for r in reqs:
    rows.append({"group": r["group"],
                 "a_blocked": not mech_a(r),
                 "b_blocked": not mech_b(r)})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "per_request_actg.csv", index=False)

print("\n" + "=" * 62)
s = df.groupby("group").agg(n=("a_blocked", "size"),
                            A=("a_blocked", "mean"),
                            B=("b_blocked", "mean"))
s["A"] = (100 * s.A).round(0).astype(int).astype(str) + "%"
s["B"] = (100 * s.B).round(0).astype(int).astype(str) + "%"
print(s.to_string())
print("=" * 62)
print("\nwrote spec/actg_cox_v1.yaml, data/requests_actg.json,")
print("      results/per_request_actg.csv")
