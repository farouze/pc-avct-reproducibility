"""Matched conventional-versus-attractor AuroraBP feature-family ablation."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ttest_1samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

import legacy_helpers as legacy

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["PCAVCT_DATA_DIR"]).resolve()
OUT = Path(os.environ.get(
    "PCAVCT_ABLATION_OUTPUT_DIR",
    str(HERE.parent / "runs" / "feature-family-ablation"),
)).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
WINDOW_SECONDS = 10
V26_TOP_K = 24
V26_CAL_PATTERN = r"^\s*calibration\s+start"
V26_ANY_CAL_PATTERN = r"\bcalibration\b"
RAW = legacy.RAW
PC = legacy.PC
DELTA_RAW = legacy.DELTA_RAW
RELATIVE_RAW = legacy.RELATIVE_RAW
V12_PARAMS = legacy.V12_PARAMS
V10_ALPHA_GRID = legacy.V10_ALPHA_GRID
V10_THRESHOLD_GRID = legacy.V10_THRESHOLD_GRID
exec(compile((HERE / "original_v26_functions.py").read_text(),
             str(HERE / "original_v26_functions.py"), "exec"))

legacy.UCI_CACHE = ROOT / "uci_pc_features.csv"
aurora_path = ROOT / "aurora_raw_avct_features.csv"
expected_hashes = json.loads(
    (HERE.parent / "docs" / "reference_run_manifest.json").read_text()
)["input_sha256"]
for path in [aurora_path, legacy.UCI_CACHE]:
    if not path.is_file():
        raise ValueError(f"Missing {path}; see data/README.md.")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_hashes[path.name]:
        raise ValueError(f"Input hash mismatch for {path.name}.")
aurora_features = pd.read_csv(aurora_path)
uci_prior_models, uci_prior_features = legacy.fit_uci_priors()
frames = {dose: v26_prepare_dose(dose)[0] for dose in [1, 2, 3]}
common_keys = set.intersection(*(set(v26_row_key(frame)) for frame in frames.values()))
frame = frames[3]
frame = frame.loc[[key in common_keys for key in v26_row_key(frame)]].copy()
frame = frame.loc[~frame.v26_is_calibration_labeled].reset_index(drop=True)
assert frame.subject_id.nunique() == 653 and len(frame) == 7879

context = [
    "context_optical_quality", "context_posture_seated",
    "context_activity_post_exercise",
]
standard_names = [
    "sigma_m", "skew_m", "kurtosis_m", "spectral_entropy",
    "ppg_quality", "pat_seconds",
]
nonlinear_names = [
    "sample_entropy", "permutation_entropy", "recurrence_rate",
    "determinism", "rqa_entropy", "csi",
]

def change_features(names):
    raw = ["raw_" + name for name in names]
    return ["delta_" + name for name in raw] + ["relative_" + name for name in raw]

arms = {
    "context_only": context,
    "standard_ppg": context + change_features(standard_names),
    "attractor_augmented": context + change_features(standard_names + nonlinear_names),
}

records = []
models = []
for repeat in range(3):
    for fold, train, test in v26_participant_folds(frame, 5, repeat):
        for target in ["sbp", "dbp"]:
            fit = train.dropna(subset=["delta_" + target])
            for arm, candidates in arms.items():
                features = ["baseline_" + target, "elapsed_seconds", "log_elapsed_days",
                            "session_elapsed_seconds", "log_session_elapsed_days"]
                features += candidates
                features = [name for name in dict.fromkeys(features)
                            if name in fit and fit[name].notna().any()
                            and fit[name].nunique(dropna=True) > 1]
                model = make_model_v17(V12_PARAMS)
                model.fit(
                    fit[features], fit["delta_" + target],
                    histgradientboostingregressor__sample_weight=abl_weights(
                        fit, "fst_band", False),
                )
                gate = abl_fit_gate(
                    fit, target, "baseline_" + target,
                    model.predict(fit[features]), "fst_band", False,
                )
                prediction, active = abl_apply_gate(
                    test, "baseline_" + target, model.predict(test[features]),
                    gate, "fst_band",
                )
                result = test[["subject_id", "_source_order", target]].copy()
                result.columns = ["subject_id", "source_order", "observed"]
                result["prediction"] = prediction
                result["repeat"] = repeat + 1
                result["fold"] = fold
                result["target"] = target
                result["arm"] = arm
                records.append(result)
                models.append({
                    "repeat": repeat + 1, "fold": fold, "target": target,
                    "arm": arm, "features": features, "gate": gate,
                    "gate_active_fraction": float(np.mean(active)),
                })
        print(f"repeat {repeat + 1} fold {fold} complete", flush=True)

predictions = pd.concat(records, ignore_index=True)
predictions["ae"] = abs(predictions.prediction - predictions.observed)
participant_repeat = (
    predictions.groupby(["repeat", "target", "arm", "subject_id"], observed=True).ae
    .mean().reset_index()
)
participant = (
    participant_repeat.groupby(["target", "arm", "subject_id"], observed=True).ae
    .mean().reset_index()
)

def bootstrap_mean(values, confidence=0.95, n_resamples=10000, seed=20260910):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 500):
        size = min(500, n_resamples - start)
        means[start:start + size] = values[
            rng.integers(0, len(values), size=(size, len(values)))
        ].mean(axis=1)
    alpha = (1 - confidence) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]

def sign_flip(values, n_resamples=99999, seed=20260910):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_resamples, 500):
        size = min(500, n_resamples - start)
        null = (rng.choice([-1, 1], size=(size, len(values))) * values).mean(axis=1)
        extreme += int((abs(null) >= observed - 1e-14).sum())
    return (extreme + 1) / (n_resamples + 1)

summary = []
contrasts = []
for target in ["sbp", "dbp"]:
    wide = participant[participant.target == target].pivot(
        index="subject_id", columns="arm", values="ae")
    for arm in arms:
        values = wide[arm].to_numpy()
        summary.append({
            "target": target, "arm": arm, "n": len(values),
            "mae": float(values.mean()), "ci95": bootstrap_mean(values),
        })
    for reference, augmented, label in [
        ("context_only", "standard_ppg", "standard_ppg_vs_context"),
        ("standard_ppg", "attractor_augmented", "attractor_vs_standard"),
    ]:
        gain = (wide[reference] - wide[augmented]).to_numpy()
        contrasts.append({
            "target": target, "contrast": label, "n": len(gain),
            "gain": float(gain.mean()), "ci95": bootstrap_mean(gain),
            "paired_signflip_p": sign_flip(gain),
            "paired_t_p": float(ttest_1samp(gain, 0).pvalue),
        })

# Holm adjustment is limited to the two endpoint tests for the prespecified
# attractor-versus-standard contrast within this secondary analysis.
indices = [i for i, row in enumerate(contrasts)
           if row["contrast"] == "attractor_vs_standard"]
pvalues = np.array([contrasts[i]["paired_signflip_p"] for i in indices])
order = np.argsort(pvalues, kind="stable")
adjusted = np.maximum.accumulate(
    np.minimum(1, pvalues[order] * np.arange(len(pvalues), 0, -1)))
holm = np.empty(len(pvalues))
holm[order] = adjusted
for i, value in zip(indices, holm):
    contrasts[i]["paired_signflip_p_holm"] = float(value)

aggregate = {
    "analysis": "Secondary matched feature-family ablation",
    "scope": (
        "Three repeats of five-fold participant CV on the strict AuroraBP cohort. "
        "Fixed feature families, estimator, weights, and training-stage gate; no UCI prior."
    ),
    "feature_families": {
        "context": context,
        "standard_ppg": standard_names,
        "nonlinear_attractor_augmentation": nonlinear_names,
    },
    "summary": summary,
    "contrasts": contrasts,
}
(OUT / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
(OUT / "selected_models.json").write_text(json.dumps(models, indent=2) + "\n")
predictions.to_csv(OUT / "row_predictions.csv", index=False)
participant_repeat.to_csv(OUT / "participant_repeat_errors.csv", index=False)
(OUT / "manifest.json").write_text(json.dumps({
    "python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "scipy": scipy.__version__,
    "sklearn": sklearn.__version__, "participants": 653, "rows": 7879,
    "repeats": 3, "folds": 5, "seed": SEED,
    "input_sha256": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [aurora_path, legacy.UCI_CACHE]
    },
}, indent=2) + "\n")
print(json.dumps(aggregate, indent=2))
