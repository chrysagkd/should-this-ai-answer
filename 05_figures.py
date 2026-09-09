from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "results" / "per_request.csv")

GROUPS = ["clean", "missing_input", "unit_mismatch",
          "coding_mismatch", "endpoint_mismatch", "out_of_population"]
LABEL = {
    "clean": "Clean requests\n(cost, not benefit)",
    "missing_input": "Required input\nabsent",
    "unit_mismatch": "Unit\nmismatch",
    "coding_mismatch": "Coding-scheme\nmismatch",
    "endpoint_mismatch": "Endpoint\nnot modelled",
    "out_of_population": "Outside development\npopulation",
}

a = [100 * df[df.group == g].a_blocked.mean() for g in GROUPS]
b = [100 * df[df.group == g].b_blocked.mean() for g in GROUPS]

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(GROUPS))
h = 0.36

ax.barh(y + h/2, a, h, label="Model-derived criterion",
        color="#B4B2A9", edgecolor="none")
ax.barh(y - h/2, b, h, label="Supervisory layer",
        color="#9E1D48", edgecolor="none")

for i, (va, vb) in enumerate(zip(a, b)):
    ax.text(va + 1.5, i + h/2, f"{va:.0f}%", va="center", fontsize=10)
    ax.text(vb + 1.5, i - h/2, f"{vb:.0f}%", va="center", fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels([LABEL[g] for g in GROUPS], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("Requests withheld (%)", fontsize=11)
ax.set_title("Withholding by release condition (n = 100 per class)",
             fontsize=12, pad=14)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0.5, color="#888780", lw=0.8, ls="--")

plt.tight_layout()
plt.savefig(ROOT / "results" / "fig1_detection.png", dpi=200)
print("wrote results/fig1_detection.png")
