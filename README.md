# PC-AVCT reproducibility

Code and provenance for the V38 **model-analysis rerun** of PC-AVCT, with the two author-supplied development notebooks preserved as source-only copies. The reference analysis uses 653 AuroraBP participants, 7,879 strict-endpoint measurements, and three repeats of five-fold participant cross-validation.

The primary wearable-state results are 7.697 mmHg SBP and 6.139 mmHg DBP participant MAE. Paired gains over the shared model without added context are 0.436 and 0.126 mmHg, with Holm-adjusted paired sign-flip p values of 0.00002 and 0.00013. See [aggregate statistics](results/aggregate/statistics.json) for unrounded values and intervals.

## Quick start

Use Python **3.11.4**, matching the recorded reference environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python scripts/reproduce.py --data-dir /path/to/local/feature_banks --mode check
python scripts/reproduce.py --data-dir /path/to/local/feature_banks --mode smoke --output-dir runs/smoke
python scripts/reproduce.py --data-dir /path/to/local/feature_banks --mode full --output-dir runs/full
```

Run from the repository root. Smoke mode fits the first outer fold for both endpoints and all four context arms; it is an execution check, not an inferential result. Full mode performs 180 AuroraBP fits (120 context-arm fits and 60 additional dose fits), then regenerates participant-level contrasts, bootstrap intervals, paired tests and Holm adjustment. Output directories must be new to avoid replacing earlier runs.

Data are not included. See [input requirements](data/README.md). Input hashes are checked before fitting. All participant-level outputs go to ignored `runs/` directories; only aggregate results are tracked.

## What the notebooks establish

- [LEAN V26/V28/V29/V30](notebooks/AVCT_Aurora_LEAN_V26_V28_V29_V30.ipynb) contains the V26 primary method, V28 controls, V29 ensemble and V30 experimental code. Its stored V26 aggregate outputs in the original attachment match the earlier paper. All 14 extracted V26 helper functions match the V38 rerun by syntax-tree comparison.
- [Monk/Aurora Sub5 notebook](notebooks/AVCT_Monk_Skin_Tone_Test_Aurora_PCAVCT_Sub5_Target.ipynb) contains separate Monk/Fitzpatrick analyses and an exploratory under-5-mmHg search. It is development material, not an interchangeable implementation of the V26 primary endpoint.

Notebook source cells are unchanged. Saved outputs, execution counts and incidental metadata were removed before repository inclusion. Original and sanitized file hashes are recorded in [notebook provenance](docs/notebook_provenance.json). Historical Colab setup cells contain their original paths and package-install commands; they do not define a frozen reproduction environment. Use the portable runner for the reference analysis.

## Validation

```sh
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Tests cover calibration-block exclusion and dose construction, participant-disjoint folds, source-function provenance, and paired inference on controlled examples. The release check rejects notebook outputs, likely credentials, and tracked participant-data files. GitHub Actions runs these checks without restricted datasets. Passing CI does **not** mean the clinical analysis ran on GitHub.

## Scope and interpretation

The reference rerun begins with hashed **feature banks**, not raw waveforms. Small differences from the old notebook's numerical outputs remain; exact old-run recovery has not been demonstrated. Reproducing the same functions does not establish the original environment or extraction history. The restored Holm correction applies only to the two retrospectively designated primary endpoints and does not erase development-selection effects or shared-training dependence.

The UCI prior uses corrected features and was selected in 26 of 30 endpoint/fold core models. “No added Fitzpatrick terms” is therefore not a claim of complete pigmentation independence. The primary AuroraBP quality feature is the filtered PPG Welch power ratio over 0.5–5 Hz versus 0.1–15 Hz; it differs from dataset-supplied optical quality and pulse-fusion weights. The recorded waveform audit supports 500 Hz.

## Citation, archival release and reuse

[CITATION.cff](CITATION.cff) records the authors and version. Both authors approved the revised manuscript and public release of this repository on 2026-09-10. A GitHub repository URL is not a DOI. No software DOI has been assigned or invented. Connect a real archival release (for example, Zenodo) and add its issued DOI before citing it in the paper. See [release checklist](docs/RELEASE.md).

No open-source license has been selected on behalf of both authors. Until they choose one, this repository does not grant an explicit reuse license. Third-party datasets retain their own access and licensing conditions.
