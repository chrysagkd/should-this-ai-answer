from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260903)
N = 100

with open(ROOT / "models" / "gbsg2_cox_v1.pkl", "rb") as f:
    B = pickle.load(f)

X_dev, X_out, FEATURES = B["X_dev"], B["X_out"], B["features"]

# Crosswalk from the model's three-tier grading to a four-tier scheme.
# DESIGN CHOICE, not data: three-tier II spans what a four-tier scheme
# splits into G2/G3. A receiving system that reads the label ordinally
# therefore assigns a different number to the same tumour.
THREE_TO_FOUR = {"I": "G1", "II": "G3", "III": "G4"}


def base(pid, row, group):
    """A well-formed request for a real patient."""
    return {
        "request_id": f"{group}_{pid}",
        "group": group,
        "patient": {
            "age": float(row.age),
            "tsize": float(row.tsize),
            "pnodes": float(row.pnodes),
            "tgrade": str(row.tgrade),
            "progrec": float(row.progrec),
        },
        "requested_endpoint": "recurrence_free_survival",
        "units": {"tsize": "mm"},
        "coding": {"tgrade": "three_tier"},
    }


requests = []

#1. clean
idx = RNG.choice(X_dev.index, N, replace=False)
for pid in idx:
    requests.append(base(pid, X_dev.loc[pid], "clean"))

#2. required value absent
# tgrade removed. The prediction path imputes it silently (see pipeline.py).
idx = RNG.choice(X_dev.index, N, replace=False)
for pid in idx:
    r = base(pid, X_dev.loc[pid], "missing_input")
    r["patient"]["tgrade"] = None
    requests.append(r)

#3. unit mismatch
# tsize in cm/ the model expects mm.
idx = RNG.choice(X_dev.index, N, replace=False)
for pid in idx:
    r = base(pid, X_dev.loc[pid], "unit_mismatch")
    r["patient"]["tsize"] = r["patient"]["tsize"] / 10.0
    r["units"]["tsize"] = "cm"
    requests.append(r)

#4. coding mismatch
idx = RNG.choice(X_dev.index, N, replace=False)
for pid in idx:
    r = base(pid, X_dev.loc[pid], "coding_mismatch")
    r["patient"]["tgrade"] = THREE_TO_FOUR[r["patient"]["tgrade"]]
    r["coding"]["tgrade"] = "four_tier"
    requests.append(r)

#5. endpoint mismatch
# Patient and values are entirely correct. Only the question is wrong.
idx = RNG.choice(X_dev.index, N, replace=False)
for pid in idx:
    r = base(pid, X_dev.loc[pid], "endpoint_mismatch")
    r["requested_endpoint"] = "overall_survival"
    requests.append(r)

#6. out of population
# Real patients aged under 45, held out in 01. Nothing edited.
idx = RNG.choice(X_out.index, N, replace=False)
for pid in idx:
    requests.append(base(pid, X_out.loc[pid], "out_of_population"))

with open(ROOT / "data" / "requests.json", "w") as f:
    json.dump(requests, f, indent=1)

df = pd.DataFrame(requests)
print(df.group.value_counts().to_string())
print(f"\ntotal: {len(requests)}")
print(f"\nexample (unit_mismatch):")
print(json.dumps([r for r in requests if r["group"] == "unit_mismatch"][0],
                 indent=2))
