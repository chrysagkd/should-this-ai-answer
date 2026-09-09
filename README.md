# Should this AI answer?

## An independent supervisory layer for prognostic decision support

A prognostic model returns an estimate for whatever arrives at its input. This repository separates two functions that are normally treated as the same one:

* **generating** a prediction &
* **deciding whether it may be released**.

The release decision is taken by a deterministic layer that reads a machine-readable specification of the model's operating conditions. It is compared against a criterion derived from the model itself.

> **Key finding.** A criterion derived from the model missed 69–93% of requests containing a declared input or intended-use violation, depending on class. In GBSG2, 91% of requests asking for an endpoint the model does not estimate were answered normally. In a cohort recording two real endpoints on the same patients, the two endpoint models assigned **different risk deciles to 34% of patients** — with the displayed number identical.

This is a proof of concept. The prognostic models are substrates. No claim is made about their predictive performance.

---

## Data

Both cohorts are distributed with `scikit-survival`. Nothing is downloaded and nothing is synthesised - every request is built from a real patient record.

| Cohort                                         |    n | Development population | Note                                                                                     |
| ---------------------------------------------- | ---: | ---------------------- | ---------------------------------------------------------------------------------------- |
| **GBSG2** — German Breast Cancer Study Group 2 |  686 | Ages 45–85 (n = 555)   | The 131 patients below 45 are held out and reused as genuine out-of-population requests. |
| **ACTG** — AIDS clinical trial cohort          | 1151 | Ages 30–55 (n = 950)   | Records **two real endpoints on the same patients**: progression to AIDS and death.      |

---

## Release conditions tested

Six classes, 100 requests each.

| Class                          | What arrives                                                     |
| ------------------------------ | ---------------------------------------------------------------- |
| Clean                          | A well-formed request                                            |
| Required input absent          | A value is missing and the prediction path imputes it silently   |
| Unit mismatch                  | Tumour size in cm where mm is expected                           |
| Coding-scheme mismatch         | Grade in a different scheme. Karnofsky/ECOG in the second cohort |
| Endpoint not modelled          | A different endpoint is requested                                |
| Outside development population | A real patient outside the stated age range                      |

---

## Mechanisms compared

### A — Model-derived criterion

Bootstrap dispersion of the risk score using 200 refits, combined with Mahalanobis distance of the covariate vector.

The criterion is calibrated to a 5% block rate on clean requests. This operating point is stated in advance and is not tuned to the evaluation result.

### B — Supervisory layer

A deterministic layer reads preconditions from the machine-readable specification:

* required inputs are present;
* declared units match;
* the coding scheme matches;
* the requested endpoint is the modelled endpoint; and
* the patient belongs to the development population.

If a precondition fails, the layer returns the failing condition instead of an estimate.

The supervisory layer is independent of the model's parameters and inference pathway. It knows nothing about the model's internal computations.

---

## Results — GBSG2

| Class                          | A withheld | B withheld | Released by A                         |
| ------------------------------ | ---------: | ---------: | ------------------------------------- |
| Clean *(cost, not benefit)*    |         5% |         0% | —                                     |
| Required input absent          |         7% |       100% | 93%                                   |
| Unit mismatch                  |        11% |       100% | 89%, of which 74 had a shifted decile |
| Coding-scheme mismatch         |        31% |       100% | 69%                                   |
| Endpoint not modelled          |         9% |       100% | 91%                                   |
| Outside development population |        23% |       100% | 77%                                   |

B reaching 100% is a property of the specification, not an empirical performance finding: the preconditions are written to identify these conditions. The empirical findings are the behaviour of the model-derived criterion and the consequences of requests that it releases.

---

## Results — ACTG

The same specification **schema** is reused unchanged, while the cohort-specific operating conditions are declared afresh for this cohort.

B again withholds every declared violation at no cost on clean requests. A behaves differently across cohorts:

* **80%** withholding for coding-scheme mismatch;
* **59%** withholding for out-of-population requests.

The model-derived criterion changes substantially across cohorts, whereas the declared preconditions retain the same decision semantics.

### Endpoint disagreement

The ACTG cohort records two real endpoints on the same patients. Therefore, the cost of misreading the endpoint can be measured rather than only detected.

The two endpoint models assigned different risk deciles to **34% of patients**, while the displayed patient number remained identical.

The death model is fitted on 16 events and is weak. The 34% result concerns **rank agreement**, not predictive performance.

---

## Reproducing

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the main evaluation:

```bash
python src/01_fit_models.py
python src/02_build_requests.py
python src/04_run_eval.py
python src/05_figures.py
python src/06_second_cohort.py
```

The scripts perform the following steps:

```text
01_fit_models.py       Fit the substrate model and derive its specification
02_build_requests.py   Build 600 requests across six classes
04_run_eval.py         Run both release mechanisms and build the results table
05_figures.py          Generate the main comparison figure
06_second_cohort.py    Run the ACTG evaluation and endpoint-disagreement analysis
```

`pipeline.py` and `03_mechanisms.py` are modules imported by the scripts above and are not required to be run directly. Running `03_mechanisms.py` independently prints a per-class sanity check.

The evaluation writes the following outputs to `results/`:

* per-request decisions;
* the aggregate detection table;
* the main comparison figure; and
* endpoint-disagreement results for the second cohort.

All randomness is seeded.

---

## Repository contents

| File or directory          | Purpose                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `src/01_fit_models.py`     | Fits the substrate model and derives the operating specification from the development data |
| `src/02_build_requests.py` | Builds 600 requests across six classes from real patient records                           |
| `src/pipeline.py`          | Prediction path under test, including silent imputation                                    |
| `src/03_mechanisms.py`     | Implements the two release mechanisms                                                      |
| `src/04_run_eval.py`       | Runs both mechanisms and writes the detection table                                        |
| `src/05_figures.py`        | Generates the main comparison figure                                                       |
| `src/06_second_cohort.py`  | Runs the second-cohort and endpoint-disagreement analysis                                  |
| `spec/`                    | Machine-readable operating specifications in YAML                                          |
| `results/`                 | Detection tables, per-request records and figures                                          |
| `requirements.txt`         | Python dependencies                                                                        |
| `LICENSE`                  | MIT licence                                                                                |

---

## Limitations

* The substrate models are used only to provide a prediction pathway. No claim is made about their predictive performance.
* In GBSG2, the three-tier to four-tier grade crosswalk is a design choice and is stated in the code.
* In ACTG, the Karnofsky–ECOG crosswalk is a published clinical mapping.
* The ACTG death model is fitted on few events.
* “Independent” means independent of the model's parameters and inference pathway, not independent of the party that authors the specification.
* The specification describes declared operating conditions; it does not establish that those conditions are clinically sufficient.
* No clinical validation was performed.
* This is a mechanism prototype, not a deployable clinical system.

---

## Licence

MIT — see [`LICENSE`](LICENSE).
