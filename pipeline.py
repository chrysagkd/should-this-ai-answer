import numpy as np

FOUR_TIER = ["G1", "G2", "G3", "G4"]


def prepare_input(request, bundle):
    
    p = request["patient"]
    order = bundle["tgrade_order"]
    med = bundle["dev_median"]
    notes = []

    # tgrade=ordinal position within the scheme it arrived in
    g = p["tgrade"]
    if g is None:
        grade = med["tgrade"]
        notes.append("tgrade imputed with development median")
    elif g in order:
        grade = float(order.index(g))
    elif g in FOUR_TIER:
        # read ordinally in its own scheme - no conversion applied
        grade = float(FOUR_TIER.index(g))
        notes.append("tgrade read ordinally from a different scheme")
    else:
        grade = med["tgrade"]
        notes.append("tgrade unrecognised, imputed")

    #everything else used as supplied/ declared units ignored
    vec = np.array([
        p["age"],
        p["tsize"],
        p["pnodes"],
        grade,
        p["progrec"],
    ], dtype=float)

    return vec, notes


def predict_decile(vec, bundle):
    """Risk decile 1-10, the form a clinician would be shown."""
    risk = bundle["model"].predict(vec.reshape(1, -1))[0]
    return int(np.searchsorted(bundle["decile_edges"], risk) + 1), risk
