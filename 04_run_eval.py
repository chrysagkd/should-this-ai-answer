from pathlib import Path
import json
import pickle
import importlib
import numpy as np
import pandas as pd

from pipeline import prepare_input, predict_decile

mech = importlib.import_module("03_mechanisms")

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)

GROUPS = ["clean", "missing_input", "unit_mismatch",
          "coding_mismatch", "endpoint_mismatch", "out_of_population"]

LABEL = {
    "clean": "Clean requests (valid, wrongly blocked)",
    "missing_input": "Required input absent",
    "unit_mismatch": "Unit mismatch",
    "coding_mismatch": "Coding-scheme mismatch",
    "endpoint_mismatch": "Endpoint not modelled",
    "out_of_population": "Outside development population",
}

# classes where the same patient also has a well-formed counterpart, so a decile shift is defined
SHIFT_DEFINED = {"missing_input", "unit_mismatch", "coding_mismatch"}

bundle = pickle.load(open(ROOT / "models" / "gbsg2_cox_v1.pkl", "rb"))
requests = json.load(open(ROOT / "data" / "requests.json"))
clean = [r for r in requests if r["group"] == "clean"]

print("building mechanism A")
A = mech.ModelDerived(bundle)
A.calibrate(clean)
B = mech.SupervisoryLayer(ROOT / "spec" / "gbsg2_cox_v1.yaml")


def clean_counterpart(r):
    """The same request as it would have arrived well-formed."""
    c = json.loads(json.dumps(r))
    p, o = c["patient"], r["patient"]
    if r["group"] == "unit_mismatch":
        p["tsize"] = o["tsize"] * 10.0
        c["units"]["tsize"] = "mm"
    elif r["group"] == "coding_mismatch":
        inv = {v: k for k, v in
               {"I": "G1", "II": "G3", "III": "G4"}.items()}
        p["tgrade"] = inv[o["tgrade"]]
        c["coding"]["tgrade"] = "three_tier"
    elif r["group"] == "missing_input":
        # original grade is not recoverable from the request itsel look it up from the development record
        pid = int(r["request_id"].split("_")[-1])
        p["tgrade"] = str(bundle["X_dev"].loc[pid, "tgrade"])
    return c


rows = []
print("\nevaluating 600 requests")
for r in requests:
    vec, notes = prepare_input(r, bundle)
    dec, risk = predict_decile(vec, bundle)

    a_ok, a_why = A.check(r)
    b_ok, b_why = B.check(r)

    shift = np.nan
    if r["group"] in SHIFT_DEFINED:
        cvec, _ = prepare_input(clean_counterpart(r), bundle)
        cdec, _ = predict_decile(cvec, bundle)
        shift = abs(dec - cdec)

    rows.append({
        "request_id": r["request_id"],
        "group": r["group"],
        "decile": dec,
        "decile_shift": shift,
        "a_blocked": not a_ok,
        "b_blocked": not b_ok,
        "a_reason": a_why,
        "b_reason": b_why,
        "silent_notes": "; ".join(notes),
    })

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "per_request.csv", index=False)

# summary table
def ci(k, n):
    """Wilson 95% interval, as a percentage."""
    if n == 0:
        return (np.nan, np.nan)
    z, p = 1.96, k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


out = []
for g in GROUPS:
    s = df[df.group == g]
    n = len(s)
    ka, kb = int(s.a_blocked.sum()), int(s.b_blocked.sum())
    la, ua = ci(ka, n)
    lb, ub = ci(kb, n)

    if g in SHIFT_DEFINED:
        sh = s.decile_shift
        shift_txt = (f"{sh.median():.1f} "
                     f"(IQR {sh.quantile(.25):.1f}-{sh.quantile(.75):.1f})")
        # silent failure: passed both checks AND moved the decile
        silent = int(((~s.a_blocked) & (~s.b_blocked)
                      & (s.decile_shift > 0)).sum())
        silent_txt = f"{100*silent/n:.0f}%"
    elif g == "endpoint_mismatch":
        shift_txt = "0 (output identical)"
        silent_txt = "n/a"
    else:
        shift_txt = "n/a"
        silent_txt = "n/a"

    out.append({
        "Class": LABEL[g],
        "n": n,
        "A blocked": f"{100*ka/n:.0f}% ({la:.0f}-{ua:.0f})",
        "B blocked": f"{100*kb/n:.0f}% ({lb:.0f}-{ub:.0f})",
        "Median decile shift": shift_txt,
        "Silent (passed both, shifted)": silent_txt,
    })

summary = pd.DataFrame(out)
summary.to_csv(ROOT / "results" / "detection_table.csv", index=False)

print("\n" + "=" * 100)
print(summary.to_string(index=False))
print("=" * 100)
print("\nA = model-derived criterion, calibrated to a 5% block rate on "
      "clean requests")
print("B = supervisory layer reading the operating specification")
print("Percentages with Wilson 95% intervals. n = 100 per class.")
print(f"\nwrote results/detection_table.csv and results/per_request.csv")
