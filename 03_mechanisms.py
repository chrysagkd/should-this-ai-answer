from pathlib import Path
import pickle
import numpy as np
import yaml
from scipy.spatial.distance import mahalanobis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from pipeline import prepare_input

ROOT = Path(__file__).resolve().parent
N_BOOT = 200
RNG = np.random.default_rng(20260903)


# mechanism A
class ModelDerived:
    """Release criterion built from the model and its development data."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.mu = bundle["dev_mean"]
        self.vi = np.linalg.inv(bundle["dev_cov"])
        self.boot = self._fit_bootstrap(bundle)
        self.threshold = None  # set by calibrate()

    def _fit_bootstrap(self, bundle):
        from sksurv.datasets import load_gbsg2  # noqa
        import pandas as pd

        X, y = bundle["X_dev"], bundle["y_dev"]
        order = bundle["tgrade_order"]
        Xn = X.copy()
        Xn["tgrade"] = pd.Categorical(
            Xn["tgrade"], categories=order, ordered=True
        ).codes.astype(float)
        Xn = Xn.astype(float).values

        models, n = [], len(Xn)
        for _ in range(N_BOOT):
            idx = RNG.choice(n, n, replace=True)
            try:
                models.append(
                    CoxPHSurvivalAnalysis().fit(Xn[idx], y[idx])
                )
            except Exception:
                continue
        print(f"  bootstrap models fitted: {len(models)}/{N_BOOT}")
        return models

    def _scores(self, vec):
        risks = np.array([m.predict(vec.reshape(1, -1))[0]
                          for m in self.boot])
        return risks.std(), mahalanobis(vec, self.mu, self.vi)

    def calibrate(self, clean_requests):
        """Set the operating point at a 5% block rate on clean requests."""
        d, m = [], []
        for r in clean_requests:
            vec, _ = prepare_input(r, self.bundle)
            a, b = self._scores(vec)
            d.append(a)
            m.append(b)
        self.d95 = np.percentile(d, 95)
        self.m95 = np.percentile(m, 95)
        combined = np.maximum(np.array(d) / self.d95, np.array(m) / self.m95)
        self.threshold = np.percentile(combined, 95)
        print(f"  calibrated: dispersion p95={self.d95:.3f}  "
              f"mahalanobis p95={self.m95:.3f}  threshold={self.threshold:.3f}")

    def check(self, request):
        vec, _ = prepare_input(request, self.bundle)
        d, m = self._scores(vec)
        c = max(d / self.d95, m / self.m95)
        if c > self.threshold:
            which = "dispersion" if d / self.d95 > m / self.m95 else "distance"
            return False, f"{which} above operating point"
        return True, ""


#mechanism B
class SupervisoryLayer:
    """Deterministic preconditions read from the operating specification."""

    def __init__(self, spec_path):
        with open(spec_path) as f:
            self.spec = yaml.safe_load(f)

    def check(self, request):
        s, p = self.spec, request["patient"]

        # 1. required inputs present
        for k in s["required_inputs"]:
            if p.get(k) is None:
                return False, f"required input absent: {k}"

        # 2. declared units match
        for k, u in s["input_units"].items():
            if k in request["units"] and request["units"][k] != u:
                return False, f"unit mismatch: {k} supplied in " \
                              f"{request['units'][k]}, expected {u}"

        # 3. declared coding scheme matches
        for k, c in s["coding"].items():
            if k in request["coding"] and request["coding"][k] != c["scheme"]:
                return False, f"coding-scheme mismatch: {k} supplied as " \
                              f"{request['coding'][k]}, expected {c['scheme']}"

        # 4. requested endpoint is the one modelled
        if request["requested_endpoint"] != s["endpoint"]["modelled"]:
            return False, f"endpoint not modelled: " \
                          f"{request['requested_endpoint']}"

        # 5. patient within the development population
        for k, rng in s["population"].items():
            if not (rng["min"] <= p[k] <= rng["max"]):
                return False, f"outside development population: " \
                              f"{k}={p[k]} not in [{rng['min']}, {rng['max']}]"

        return True, ""


def load_bundle():
    with open(ROOT / "models" / "gbsg2_cox_v1.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import json

    bundle = load_bundle()
    reqs = json.load(open(ROOT / "data" / "requests.json"))
    clean = [r for r in reqs if r["group"] == "clean"]

    print("mechanism A")
    a = ModelDerived(bundle)
    a.calibrate(clean)

    print("\nmechanism B")
    b = SupervisoryLayer(ROOT / "spec" / "gbsg2_cox_v1.yaml")
    print(f"  {len(b.spec['required_inputs'])} required inputs, "
          f"5 preconditions")

    print("\nsanity check on one request per group:")
    seen = set()
    for r in reqs:
        if r["group"] in seen:
            continue
        seen.add(r["group"])
        ra, sa = a.check(r)
        rb, sb = b.check(r)
        print(f"  {r['group']:<20} A={'pass' if ra else 'BLOCK':<5} "
              f"B={'pass' if rb else 'BLOCK':<5} {sb}")
