from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yaml

from sksurv.datasets import load_gbsg2
from sksurv.linear_model import CoxPHSurvivalAnalysis

ROOT = Path(__file__).resolve().parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "spec").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

AGE_MIN, AGE_MAX = 45, 85
FEATURES = ["age", "tsize", "pnodes", "tgrade", "progrec"]

# load
X, y = load_gbsg2()
X = X[FEATURES].copy()

in_pop = (X.age >= AGE_MIN) & (X.age <= AGE_MAX)
X_dev, y_dev = X[in_pop], y[in_pop.values]
X_out = X[~in_pop]

print(f"development set : {len(X_dev)}")
print(f"held out (out of population): {len(X_out)}")

# encode & fit
# tgrade is ordinal (I < II < III) -> integer codes, order preserved
TGRADE_ORDER = ["I", "II", "III"]


def encode(df):
    out = df.copy()
    out["tgrade"] = pd.Categorical(
        out["tgrade"], categories=TGRADE_ORDER, ordered=True
    ).codes.astype(float)
    return out.astype(float)


Xd = encode(X_dev)
model = CoxPHSurvivalAnalysis().fit(Xd, y_dev)

risk_dev = model.predict(Xd)
print(f"\nconcordance (development, in-sample): "
      f"{model.score(Xd, y_dev):.3f}")
print("  -- reported for context only; no claim is made about "
      "predictive performance")

# decile cut points
# risk scores are converted to deciles so that 'shift in output' is
# reported on a scale a reader can interpret
decile_edges = np.quantile(risk_dev, np.linspace(0, 1, 11))[1:-1]

# operating specification
spec = {
    "model_id": "gbsg2_cox_v1",
    "substrate": "GBSG2 (German Breast Cancer Study Group 2)",
    "endpoint": {
        "modelled": "recurrence_free_survival",
        "not_valid_for": ["overall_survival", "cancer_specific_survival"],
    },
    "population": {
        "age": {"min": float(AGE_MIN), "max": float(AGE_MAX)},
    },
    "required_inputs": FEATURES,
    "input_units": {
        "tsize": "mm",
        "age": "years",
        "progrec": "fmol_per_mg",
    },
    "coding": {
        "tgrade": {"scheme": "three_tier", "allowed": TGRADE_ORDER},
    },
    # observed ranges, used by the statistical arm of mechanism A
    "observed_ranges": {
        c: {"p01": float(np.percentile(Xd[c], 1)),
            "p99": float(np.percentile(Xd[c], 99))}
        for c in FEATURES
    },
}

with open(ROOT / "spec" / "gbsg2_cox_v1.yaml", "w") as f:
    yaml.safe_dump(spec, f, sort_keys=False)

# persist
with open(ROOT / "models" / "gbsg2_cox_v1.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "features": FEATURES,
            "tgrade_order": TGRADE_ORDER,
            "X_dev": X_dev,
            "y_dev": y_dev,
            "X_out": X_out,
            "risk_dev": risk_dev,
            "decile_edges": decile_edges,
            "dev_median": Xd.median().to_dict(),
            "dev_mean": Xd.mean().values,
            "dev_cov": np.cov(Xd.values, rowvar=False),
        },
        f,
    )

print(f"\nwrote spec/gbsg2_cox_v1.yaml")
print(f"wrote models/gbsg2_cox_v1.pkl")
