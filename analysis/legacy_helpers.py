"""Reproduce V22's frozen tone-blind model and stratify its errors by FST.

No Fitzpatrick variables enter feature selection, weighting, model fitting,
gating, or prediction. Fitzpatrick is joined only after predictions are frozen.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
import warnings
import zipfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch, wiener
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


SEED = 42
WINDOW_SECONDS = 10
BOOTSTRAPS = 10_000
PERMUTATIONS = 10_000
V12_TOP_K = 24
V12_PARAMS = {
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}
V10_ALPHA_GRID = np.round(np.linspace(0, 1, 11), 2)
V10_THRESHOLD_GRID = np.array([0, 1, 2, 3, 4, 5, 7.5, 10.0], float)
PPG_FEATURES = [
    "sigma_m", "skew_m", "kurtosis_m", "sample_entropy",
    "permutation_entropy", "recurrence_rate", "determinism",
    "rqa_entropy", "csi", "spectral_entropy", "ppg_quality",
    "pat_seconds",
]
RAW = ["raw_" + name for name in PPG_FEATURES]
DELTA_RAW = ["delta_" + feature for feature in RAW]
RELATIVE_RAW = ["relative_" + feature for feature in RAW]
PC = ["pc_" + name for name in PPG_FEATURES]
DELTA_PC = ["delta_" + feature for feature in PC]
RELATIVE_PC = ["relative_" + feature for feature in PC]

ARCHIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/farouze/Downloads/AuroraBP.zip"
)
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "fixed_toneblind_fitzpatrick_results"
)
FEATURE_CACHE = OUTPUT / "aurora_raw_avct_features.csv"
UCI_DIRECTORY = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("cuffless_uci_340")
UCI_CACHE = OUTPUT / "uci_pc_features.csv"

_ZIP_PATH: str | None = None
_ZIP_HANDLE: zipfile.ZipFile | None = None
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.signal")


def fill_gaps(x, minimum_valid=0.95):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.mean() < minimum_valid or good.sum() < 2:
        return None
    if not good.all():
        indices = np.arange(len(x))
        x = np.interp(indices, indices[good], x[good])
    return x


def bandpass(x, fs, low, high):
    high = min(high, fs / 2 * 0.95)
    return sosfiltfilt(
        butter(3, [low, high], btype="bandpass", fs=fs, output="sos"), x
    )


def embed(x, m=4, tau=5):
    n = len(x) - (m - 1) * tau
    if n < 20:
        return np.empty((0, m))
    return np.column_stack([x[j * tau : j * tau + n] for j in range(m)])


def sample_entropy(x, m=2, r=0.2):
    sd = np.std(x)
    if len(x) < 50 or sd < 1e-10:
        return np.nan
    z = (x - np.mean(x)) / sd
    if len(z) > 600:
        z = z[np.linspace(0, len(z) - 1, 600).astype(int)]
    longer = embed(z, m + 1, 1)
    shorter = embed(z, m, 1)
    a_count = len(cKDTree(longer).query_pairs(r, p=np.inf, output_type="ndarray"))
    b_count = len(cKDTree(shorter).query_pairs(r, p=np.inf, output_type="ndarray"))
    return float(-np.log((a_count + 1) / (b_count + 1)))


def permutation_entropy(x, order=3):
    if len(x) < 30:
        return np.nan
    patterns = np.array([np.argsort(x[i : i + order]) for i in range(len(x) - order + 1)])
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(
        -np.sum(probabilities * np.log(probabilities + 1e-12))
        / np.log(math.factorial(order))
    )


def runs(values):
    changes = np.diff(np.r_[0, np.asarray(values, dtype=np.int8), 0])
    return np.where(changes == -1)[0] - np.where(changes == 1)[0]


def rqa(x, eps=0.20):
    sd = np.std(x)
    if sd < 1e-10:
        return (np.nan,) * 3
    embedded = embed((x - np.mean(x)) / sd)
    if len(embedded) > 300:
        embedded = embedded[np.linspace(0, len(embedded) - 1, 300).astype(int)]
    recurrence = cdist(embedded, embedded, metric="chebyshev") <= eps
    np.fill_diagonal(recurrence, False)
    total = recurrence.sum()
    rate = total / (len(recurrence) * (len(recurrence) - 1))
    lines = []
    for offset in range(-len(recurrence) + 1, len(recurrence)):
        if offset:
            lengths = runs(np.diagonal(recurrence, offset))
            lines.extend(lengths[lengths >= 2])
    if total == 0 or not lines:
        return float(rate), 0.0, 0.0
    lines = np.asarray(lines)
    determinism = lines.sum() / total
    _, counts = np.unique(lines, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    return float(rate), float(determinism), float(entropy)


def spectral(x, fs):
    frequencies, power = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    full = (frequencies >= 0.1) & (frequencies <= min(15, fs / 2 - 0.1))
    pulse = (frequencies >= 0.5) & (frequencies <= 5)
    total = power[full].sum()
    if total <= 0:
        return np.nan, np.nan
    distribution = power[full] / total
    entropy = -np.sum(distribution * np.log(distribution + 1e-12)) / np.log(
        max(len(distribution), 2)
    )
    return float(power[pulse].sum() / total), float(entropy)


def avct_features(ppg, ecg, fs, prefix, noise_ratio):
    ppg = fill_gaps(ppg)
    if ppg is None or np.std(ppg) < 1e-8:
        return None
    filtered = bandpass(ppg, fs, 0.5, 8)
    scale = max(
        abs(np.median(ppg)), np.percentile(ppg, 95) - np.percentile(ppg, 5), 1e-6
    )
    ac = filtered / scale
    embedded = embed(ac)
    rho = float(np.clip(noise_ratio, 0, 1.5))
    radius = 0.20 * np.sqrt(1 + rho**2)
    recurrence, determinism, rqa_entropy = rqa(ac, eps=radius)
    quality, spectral_entropy = spectral(filtered, fs)
    csi = (
        0.4 * determinism
        + 0.3 * (1 - np.clip(rqa_entropy / 5, 0, 1))
        + 0.3 * quality
        if np.all(np.isfinite([determinism, rqa_entropy, quality]))
        else np.nan
    )
    pat = np.nan
    if ecg is not None:
        ecg = fill_gaps(ecg)
        if ecg is not None and np.std(ecg) >= 1e-8:
            ecg_filtered = bandpass(ecg, fs, 5, min(25, fs / 2 - 1))
            if abs(np.percentile(ecg_filtered, 1)) > abs(np.percentile(ecg_filtered, 99)):
                ecg_filtered = -ecg_filtered
            peaks, _ = find_peaks(
                ecg_filtered, distance=int(0.3 * fs),
                prominence=max(0.5 * np.std(ecg_filtered), 1e-8),
            )
            derivative = np.gradient(filtered)
            values = []
            for peak in peaks:
                start = peak + int(0.08 * fs)
                stop = min(len(filtered), peak + int(0.45 * fs))
                if stop > start + 2:
                    values.append((start + np.argmax(derivative[start:stop]) - peak) / fs)
            values = [value for value in values if 0.08 <= value <= 0.45]
            pat = float(np.median(values)) if values else np.nan
    return {
        prefix + "sigma_m": float(np.std(embedded)),
        prefix + "skew_m": float(skew(embedded.ravel())),
        prefix + "kurtosis_m": float(kurtosis(embedded.ravel())),
        prefix + "sample_entropy": sample_entropy(ac, r=radius),
        prefix + "permutation_entropy": permutation_entropy(ac),
        prefix + "recurrence_rate": recurrence,
        prefix + "determinism": determinism,
        prefix + "rqa_entropy": rqa_entropy,
        prefix + "csi": csi,
        prefix + "spectral_entropy": spectral_entropy,
        prefix + "ppg_quality": quality,
        prefix + "pat_seconds": pat,
    }


def init_worker(zip_path):
    global _ZIP_PATH, _ZIP_HANDLE
    _ZIP_PATH = zip_path
    _ZIP_HANDLE = zipfile.ZipFile(zip_path)


def extract_one(record):
    member = "AuroraBP/" + str(record["waveform_file_path"]).lstrip("/")
    try:
        with _ZIP_HANDLE.open(member) as raw:
            waveform = pd.read_csv(raw, sep="\t", usecols=["t", "optical"], low_memory=False)
        time = pd.to_numeric(waveform.t, errors="coerce").to_numpy(float)
        optical = pd.to_numeric(waveform.optical, errors="coerce").to_numpy(float)
        valid = np.isfinite(time) & np.isfinite(optical)
        time, optical = time[valid], optical[valid]
        order = np.argsort(time)
        time, optical = time[order], optical[order]
        if len(optical) < 1000 or time[-1] <= time[0]:
            raise ValueError("waveform too short")
        fs = (len(time) - 1) / (time[-1] - time[0])
        needed = int(round(WINDOW_SECONDS * fs))
        if len(optical) < needed:
            raise ValueError("less than 10 seconds")
        start = (len(optical) - needed) // 2
        x = fill_gaps(optical[start : start + needed], minimum_valid=0.90)
        if x is None:
            raise ValueError("gap rejection")
        ac = x - np.median(x)
        pulse = bandpass(x, fs, 0.5, 8)
        residual = ac - pulse
        ratio = float(np.clip(np.std(residual) / max(np.std(pulse), 1e-8), 0, 1.5))
        raw_features = avct_features(x, None, fs, "raw_", ratio)
        fst = float(np.clip(record["fitzpatrick_scale"], 1, 6))
        z = (fst - 1) / 5
        baseline = np.median(x)
        centered = x - baseline
        attenuation = max(0.45, 1 - 0.55 * z)
        restored = centered / attenuation
        kernel = max(5, int(round(0.05 * fs)))
        kernel += 1 - kernel % 2
        kernel = min(kernel, 21)
        denoised = np.asarray(wiener(restored, mysize=kernel), float)
        denoised = np.where(np.isfinite(denoised), denoised, restored)
        blend = np.clip(0.75 * z, 0, 0.75)
        corrected = baseline + (1 - blend) * restored + blend * denoised
        pc_features = avct_features(corrected, None, fs, "pc_", 0.35 * ratio)
        if raw_features is None or pc_features is None:
            raise ValueError("feature rejection")
        keep = {
            key: record[key]
            for key in [
                "pid", "phase", "measurement", "date_time", "sbp", "dbp",
                "fitzpatrick_scale", "_source_order", "duration",
                "pressure_quality", "optical_quality", "waveforms_generated",
            ]
        }
        return {
            **keep, **raw_features, **pc_features,
            "optical_noise_ratio": ratio,
            "optical_reliability": 1 / (1 + ratio**2),
        }, None
    except Exception as exc:
        return None, {"pid": record.get("pid"), "member": member, "error": str(exc)}


def extract_pc_one(record):
    """Compute only Aurora's corrected channel for enriching the raw cache."""
    member = "AuroraBP/" + str(record["waveform_file_path"]).lstrip("/")
    try:
        with _ZIP_HANDLE.open(member) as raw:
            waveform = pd.read_csv(raw, sep="\t", usecols=["t", "optical"], low_memory=False)
        time = pd.to_numeric(waveform.t, errors="coerce").to_numpy(float)
        optical = pd.to_numeric(waveform.optical, errors="coerce").to_numpy(float)
        valid = np.isfinite(time) & np.isfinite(optical)
        time, optical = time[valid], optical[valid]
        order = np.argsort(time)
        time, optical = time[order], optical[order]
        if len(optical) < 1000 or time[-1] <= time[0]:
            raise ValueError("waveform too short")
        fs = (len(time) - 1) / (time[-1] - time[0])
        needed = int(round(WINDOW_SECONDS * fs))
        if len(optical) < needed:
            raise ValueError("less than 10 seconds")
        start = (len(optical) - needed) // 2
        x = fill_gaps(optical[start : start + needed], minimum_valid=0.90)
        baseline = np.median(x)
        ac = x - baseline
        pulse = bandpass(x, fs, 0.5, 8)
        ratio = float(np.clip(np.std(ac - pulse) / max(np.std(pulse), 1e-8), 0, 1.5))
        z = (float(np.clip(record["fitzpatrick_scale"], 1, 6)) - 1) / 5
        attenuation = max(0.45, 1 - 0.55 * z)
        restored = ac / attenuation
        kernel = max(5, int(round(0.05 * fs)))
        kernel += 1 - kernel % 2
        kernel = min(kernel, 21)
        denoised = np.asarray(wiener(restored, mysize=kernel), float)
        denoised = np.where(np.isfinite(denoised), denoised, restored)
        blend = np.clip(0.75 * z, 0, 0.75)
        corrected = baseline + (1 - blend) * restored + blend * denoised
        features = avct_features(corrected, None, fs, "pc_", 0.35 * ratio)
        return {"_source_order": record["_source_order"], **features}, None
    except Exception as exc:
        return None, {"pid": record.get("pid"), "member": member, "error": str(exc)}


def load_or_extract_features():
    cached = None
    if FEATURE_CACHE.exists():
        cached = pd.read_csv(FEATURE_CACHE)
        if all(column in cached.columns for column in PC):
            print("Loading feature cache:", FEATURE_CACHE, flush=True)
            return cached
        print("Existing cache lacks the corrected channel; rebuilding it.", flush=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        participants = pd.read_csv(
            archive.open("AuroraBP/participants.tsv"), sep="\t",
            na_values=["NA", "N/A", "NaN", "nan", ""],
        )
        measurements = pd.read_csv(
            archive.open("AuroraBP/measurements_auscultatory.tsv"), sep="\t",
            na_values=["NA", "N/A", "NaN", "nan", ""], low_memory=False,
        )
    participants["fitzpatrick_scale"] = pd.to_numeric(
        participants.fitzpatrick_scale, errors="coerce"
    )
    measurements["_source_order"] = np.arange(len(measurements))
    metadata = measurements.merge(
        participants[["pid", "fitzpatrick_scale"]].drop_duplicates("pid"),
        on="pid", how="left", validate="many_to_one",
    )
    metadata = metadata.dropna(subset=["fitzpatrick_scale", "waveform_file_path"])
    records = metadata.to_dict("records")
    rows, failures = [], []
    workers = min(8, max(1, __import__("os").cpu_count() or 1))
    extractor = extract_pc_one if cached is not None else extract_one
    with Pool(workers, initializer=init_worker, initargs=(str(ARCHIVE),)) as pool:
        for index, (row, failure) in enumerate(
            pool.imap(extractor, records, chunksize=8), start=1
        ):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
            if index % 500 == 0 or index == len(records):
                print(f"Features {index}/{len(records)}; accepted {len(rows)}", flush=True)
    extracted = pd.DataFrame(rows)
    if cached is not None:
        frame = cached.merge(
            extracted, on="_source_order", how="inner", validate="one_to_one"
        )
        if len(frame) != len(cached):
            raise RuntimeError(
                f"Corrected-channel merge lost rows: {len(frame)} vs {len(cached)}"
            )
    else:
        frame = extracted
    frame.to_csv(FEATURE_CACHE, index=False)
    pd.DataFrame(failures).to_csv(OUTPUT / "feature_failures.csv", index=False)
    return frame


def build_replicate_calibrated(frame):
    source = frame.copy()
    source["sbp"] = pd.to_numeric(source.sbp, errors="coerce")
    source["dbp"] = pd.to_numeric(source.dbp, errors="coerce")
    source["_parsed_time"] = pd.to_datetime(source.date_time, errors="coerce")
    source["_is_calibration_start"] = (
        source.phase.astype(str).str.lower().eq("initial")
        & source.measurement.astype(str).str.contains(
            r"^\s*calibration\s+start", case=False, regex=True, na=False
        )
    )
    rows = []
    feature_columns = [
        column for column in source if column.startswith(("raw_", "pc_"))
    ]
    for pid, group in source.dropna(subset=["sbp", "dbp"]).groupby("pid"):
        group = group.sort_values(["_parsed_time", "_source_order"], na_position="last")
        calibrations = group[group._is_calibration_start].copy()
        if len(calibrations) < 2:
            continue
        last_order = calibrations._source_order.max()
        last_time = calibrations._parsed_time.max()
        baseline = calibrations.iloc[0].copy()
        baseline["sbp"] = calibrations.sbp.mean()
        baseline["dbp"] = calibrations.dbp.mean()
        for column in feature_columns:
            baseline[column] = pd.to_numeric(calibrations[column], errors="coerce").median()
        baseline["_source_order"] = last_order
        baseline["_parsed_time"] = last_time
        baseline["date_time"] = last_time
        baseline["measurement"] = "Calibration start replicate mean"
        baseline["subject_id"] = str(pid)
        baseline["start_seconds"] = 0.0
        future = group[
            (~group._is_calibration_start)
            & (
                (group._parsed_time > last_time)
                | (group._parsed_time.eq(last_time) & group._source_order.gt(last_order))
            )
        ]
        if future.empty:
            continue
        rows.append(baseline)
        for _, current in future.iterrows():
            item = current.copy()
            item["subject_id"] = str(pid)
            item["start_seconds"] = (current._parsed_time - last_time).total_seconds()
            rows.append(item)
    return pd.DataFrame(rows)


def build_delta_table(frame, anchor):
    rows = []
    for subject_id, group in frame.groupby("subject_id", sort=False):
        order_columns = [
            column for column in ["start_seconds", "_source_order"]
            if column in group.columns
        ]
        group = group.sort_values(order_columns, na_position="last")
        if len(group) < 2:
            continue
        session = group.iloc[0]
        for position in range(1, len(group)):
            row = group.iloc[position]
            prior = group.iloc[position - 1]
            reference = prior if anchor == "most_recent_prior" else session
            item = row.to_dict()
            item["original_index"] = group.index[position]
            row_time = pd.to_numeric(row.get("start_seconds"), errors="coerce")
            ref_time = pd.to_numeric(reference.get("start_seconds"), errors="coerce")
            session_time = pd.to_numeric(session.get("start_seconds"), errors="coerce")
            item["elapsed_seconds"] = float(row_time - ref_time)
            item["prior_gap_seconds"] = item["elapsed_seconds"]
            item["session_elapsed_seconds"] = float(row_time - session_time)
            for target in ["sbp", "dbp"]:
                current = float(row[target])
                ref_value = float(reference[target])
                session_value = float(session[target])
                item["baseline_" + target] = ref_value
                item["delta_" + target] = current - ref_value
                item["session_baseline_" + target] = session_value
                item["session_delta_" + target] = current - session_value
            for feature in RAW + PC:
                current = pd.to_numeric(row.get(feature), errors="coerce")
                initial = pd.to_numeric(reference.get(feature), errors="coerce")
                valid = np.isfinite(current) and np.isfinite(initial)
                item["delta_" + feature] = float(current - initial) if valid else np.nan
                denominator = max(abs(initial), 1e-6) if np.isfinite(initial) else np.nan
                relative = (current - initial) / denominator if valid else np.nan
                item["relative_" + feature] = (
                    float(np.clip(relative, -10, 10)) if np.isfinite(relative) else np.nan
                )
            rows.append(item)
    return pd.DataFrame(rows)


def add_context(frame):
    out = frame.copy()
    created = []
    phase = out.phase.fillna("missing").astype(str).str.strip().str.lower()
    for value in sorted(value for value in phase.unique() if value and value != "missing"):
        safe = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        column = "context_phase_" + safe
        out[column] = (phase == value).astype(float)
        created.append(column)
    label = out.measurement.fillna("").astype(str).str.lower()
    patterns = {
        "context_posture_supine": r"\bsupine\b",
        "context_posture_seated": r"\b(seated|sitting)\b",
        "context_posture_standing": r"\bstanding\b",
        "context_activity_post_exercise": r"\b(post[ -]?exercise|exercise)\b",
        "context_measurement_calibration": r"\bcalibration\b",
    }
    for column, pattern in patterns.items():
        matched = label.str.contains(pattern, regex=True, na=False)
        if matched.any():
            out[column] = matched.astype(float)
            created.append(column)
    for source, column in {
        "duration": "context_duration_seconds",
        "pressure_quality": "context_pressure_quality",
        "optical_quality": "context_optical_quality",
    }.items():
        values = pd.to_numeric(out[source], errors="coerce")
        if values.notna().any():
            out[column] = values
            created.append(column)
    return out, list(dict.fromkeys(created))


def prepare_anchor(ordered, anchor):
    frame = build_delta_table(ordered, anchor).replace([np.inf, -np.inf], np.nan)
    frame, context = add_context(frame)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["fst_band"] = pd.cut(
        frame.fitzpatrick_scale, [0, 2, 4, 6],
        labels=["Fitzpatrick I-II", "Fitzpatrick III-IV", "Fitzpatrick V-VI"],
    ).astype(str)
    frame["log_elapsed_days"] = np.log1p(frame.elapsed_seconds.clip(lower=0) / 86400)
    frame["log_session_elapsed_days"] = np.log1p(
        frame.session_elapsed_seconds.clip(lower=0) / 86400
    )
    return frame, context


def normalized_conditional_mi(frame, features, target):
    pressure = "sbp" if target.endswith("sbp") else "dbp"
    conditioning = ["baseline_" + pressure, "log_elapsed_days"]
    controls = SimpleImputer(strategy="median").fit_transform(frame[conditioning])
    controls = RobustScaler().fit_transform(controls)
    y = np.asarray(frame[target], float)
    y_residual = y - Ridge(alpha=1).fit(controls, y).predict(controls)
    x = SimpleImputer(strategy="median").fit_transform(frame[features])
    x_residual = np.empty_like(x, float)
    for column in range(x.shape[1]):
        x_residual[:, column] = x[:, column] - Ridge(alpha=1).fit(
            controls, x[:, column]
        ).predict(controls)
    values = mutual_info_regression(x_residual, y_residual, random_state=SEED)
    return values / max(float(np.max(values)), 1e-12)


def rank_features(frame, features, target):
    features = [
        feature for feature in dict.fromkeys(features)
        if feature in frame and frame[feature].notna().any()
    ]
    scores = normalized_conditional_mi(frame, features, target)
    return pd.DataFrame({"feature": features, "score": scores}).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)


def participant_weights(frame):
    visits = frame.groupby("subject_id").subject_id.transform("size").astype(float)
    weights = 1.0 / visits
    weights = np.clip(weights, np.quantile(weights, 0.02), np.quantile(weights, 0.98))
    return np.asarray(weights / weights.mean(), float)


def gate_search(frame, target, base, delta_prediction):
    baseline_mae = float(
        frame.assign(_ae=np.abs(frame[base] - frame[target]))
        .groupby("subject_id")._ae.mean().mean()
    )
    best = None
    for threshold in V10_THRESHOLD_GRID:
        active = np.abs(delta_prediction) >= threshold
        for alpha in V10_ALPHA_GRID:
            prediction = frame[base].to_numpy(float) + np.where(
                active, alpha * delta_prediction, 0.0
            )
            mae = float(
                frame.assign(_ae=np.abs(prediction - frame[target]))
                .groupby("subject_id")._ae.mean().mean()
            )
            if mae <= baseline_mae + 1e-12:
                if best is None or (mae, float(threshold)) < (best["mae"], best["threshold"]):
                    best = {"threshold": float(threshold), "alpha": float(alpha), "mae": mae}
    return best or {"threshold": float("inf"), "alpha": 0.0, "mae": baseline_mae}


def fit_toneblind(train, test, target, features):
    features = [
        feature for feature in dict.fromkeys(features)
        if feature in train and feature in test
        and train[feature].notna().any() and train[feature].nunique(dropna=True) > 1
    ]
    fit = train.dropna(subset=["delta_" + target]).copy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=300, loss="absolute_error", random_state=SEED, **V12_PARAMS
        ),
    )
    model.fit(
        fit[features], fit["delta_" + target],
        histgradientboostingregressor__sample_weight=participant_weights(fit),
    )
    development_delta = model.predict(fit[features])
    gate = gate_search(fit, target, "baseline_" + target, development_delta)
    hold_delta = model.predict(test[features])
    active = np.abs(hold_delta) >= gate["threshold"]
    prediction = test["baseline_" + target].to_numpy(float) + np.where(
        active, gate["alpha"] * hold_delta, 0.0
    )
    return prediction, features, gate


def bp_labels(arterial_pressure, fs=125):
    arterial_pressure = fill_gaps(arterial_pressure)
    if arterial_pressure is None or np.std(arterial_pressure) < 3:
        return None
    peaks, _ = find_peaks(
        arterial_pressure, distance=int(0.35 * fs), prominence=10, height=(60, 260)
    )
    if len(peaks) < 5:
        return None
    systolic = arterial_pressure[peaks]
    diastolic = np.asarray([
        np.min(arterial_pressure[start:stop])
        for start, stop in zip(peaks[:-1], peaks[1:])
    ])
    systolic = systolic[(systolic >= 70) & (systolic <= 250)]
    diastolic = diastolic[(diastolic >= 25) & (diastolic <= 150)]
    if len(systolic) < 4 or len(diastolic) < 4:
        return None
    sbp, dbp = float(np.median(systolic)), float(np.median(diastolic))
    return (sbp, dbp) if 15 <= sbp - dbp <= 150 else None


def synthetic_uci_labels(record_ids):
    identifiers = np.asarray(sorted(set(map(str, record_ids))))
    rng = np.random.default_rng(SEED + 30)
    identifiers = rng.permutation(identifiers)
    z = (np.arange(len(identifiers)) + 0.5) / max(len(identifiers), 1)
    frame = pd.DataFrame({"subject_id": identifiers, "pigmentation_index": z})
    frame["monk_synthetic"] = np.clip(np.floor(z * 10).astype(int) + 1, 1, 10)
    frame["monk_group"] = pd.cut(
        frame.monk_synthetic, [0, 3, 6, 10],
        labels=["Monk 1-3", "Monk 4-6", "Monk 7-10"],
    ).astype(str)
    return frame.set_index("subject_id").to_dict("index")


def optical_stress(ppg, z, seed, fs):
    x = np.asarray(ppg, float)
    z = float(np.clip(z, 0, 1))
    baseline = np.nanmedian(x)
    ac = x - baseline
    attenuation = max(0.2, 1 - 0.55 * z)
    noise_fraction = 0.12 * z
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_fraction * max(np.nanstd(ac), 1e-8), len(x))
    observed = baseline + attenuation * ac + noise
    restored = (observed - baseline) / attenuation
    kernel = max(5, int(round(0.05 * fs)))
    kernel += 1 - kernel % 2
    kernel = min(kernel, 21)
    denoised = np.asarray(wiener(restored, mysize=kernel), float)
    denoised = np.where(np.isfinite(denoised), denoised, restored)
    blend = np.clip(0.75 * z, 0, 0.75)
    corrected = baseline + (1 - blend) * restored + blend * denoised
    noise_ratio = noise_fraction / max(attenuation, 1e-6)
    snr_proxy = 1 / max(noise_ratio**2, 1e-6)
    reliability = snr_proxy / (1 + snr_proxy)
    return corrected, reliability, noise_ratio


def hdf_cell_key(handle):
    keys = [
        key for key in handle.keys()
        if not key.startswith("#") and isinstance(handle[key], h5py.Dataset)
    ]
    return max(keys, key=lambda key: handle[key].size)


def load_or_extract_uci():
    if UCI_CACHE.exists():
        print("Loading UCI prior cache:", UCI_CACHE, flush=True)
        return pd.read_csv(UCI_CACHE)
    parts = sorted(UCI_DIRECTORY.glob("Part_*.mat"))
    if len(parts) != 4:
        raise RuntimeError(f"Expected four official UCI Part_*.mat files in {UCI_DIRECTORY}")
    record_ids = []
    for part in parts:
        with h5py.File(part, "r") as handle:
            record_ids += [
                f"{part.stem}_{index:05d}"
                for index in range(min(handle[hdf_cell_key(handle)].size, 120))
            ]
    labels = synthetic_uci_labels(record_ids)
    rows = []
    for part_number, part in enumerate(parts, start=1):
        with h5py.File(part, "r") as handle:
            references = np.asarray(handle[hdf_cell_key(handle)]).reshape(-1)[:120]
            for record_index, reference in enumerate(references):
                if not reference:
                    continue
                signals = np.asarray(handle[reference], float).squeeze()
                if signals.ndim != 2:
                    continue
                if signals.shape[0] != 3 and signals.shape[1] == 3:
                    signals = signals.T
                if signals.shape[0] != 3:
                    continue
                ppg, arterial_pressure, ecg = signals
                record_id = f"{part.stem}_{record_index:05d}"
                label = labels[record_id]
                needed = 125 * WINDOW_SECONDS
                for window_index, start in enumerate(
                    np.arange(0, len(ppg) - needed + 1, needed)[:12]
                ):
                    bp = bp_labels(arterial_pressure[start : start + needed])
                    if bp is None:
                        continue
                    corrected, reliability, noise_ratio = optical_stress(
                        ppg[start : start + needed], label["pigmentation_index"],
                        SEED + 300000 + record_index * 100 + window_index, 125,
                    )
                    pc_features = avct_features(
                        corrected, ecg[start : start + needed], 125, "pc_",
                        0.35 * noise_ratio,
                    )
                    if pc_features is None:
                        continue
                    rows.append({
                        "subject_id": record_id,
                        "part": part.stem,
                        "start_seconds": start / 125,
                        "sbp": bp[0], "dbp": bp[1],
                        **label, **pc_features,
                        "optical_reliability": reliability,
                        "optical_noise_ratio": noise_ratio,
                    })
        print(f"UCI {part_number}/4 complete; accepted windows {len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(UCI_CACHE, index=False)
    return frame


def fit_uci_priors():
    uci_features = load_or_extract_uci()
    delta = build_delta_table(uci_features, "session_start")
    models, feature_sets = {}, {}
    for target in ["sbp", "dbp"]:
        features = DELTA_PC + RELATIVE_PC + [
            "elapsed_seconds", "optical_reliability", "optical_noise_ratio",
            "baseline_" + target,
        ]
        features = [feature for feature in features if feature in delta.columns]
        counts = delta.groupby("monk_group")["subject_id"].transform("count")
        weights = (
            len(delta) / (delta.monk_group.nunique() * counts)
            * delta.optical_reliability.clip(0.1, 1)
        )
        weights = np.asarray(weights / weights.mean())
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=300, random_state=SEED, **V12_PARAMS),
        )
        model.fit(
            delta[features], delta["delta_" + target],
            histgradientboostingregressor__sample_weight=weights,
        )
        models[target] = model
        feature_sets[target] = features
    return models, feature_sets


def add_uci_prior(frame, models, feature_sets):
    out = frame.copy()
    for target in ["sbp", "dbp"]:
        features = feature_sets[target]
        if not all(feature in out.columns for feature in features):
            missing = [feature for feature in features if feature not in out.columns]
            raise RuntimeError(f"Aurora missing UCI-prior inputs: {missing}")
        out["uci_prior_delta_" + target] = models[target].predict(out[features])
    return out


def bootstrap_mean(values, rng):
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def analyze_frozen_predictions(participant):
    bands = ["Fitzpatrick I-II", "Fitzpatrick III-IV", "Fitzpatrick V-VI"]
    summary_rows, contrast_rows, trend_rows = [], [], []
    for target_index, target in enumerate(["sbp", "dbp"]):
        metric = target + "_participant_mae"
        groups = {}
        for band_index, band in enumerate(bands):
            values = participant.loc[participant.fitzpatrick_band == band, metric].to_numpy(float)
            groups[band] = values
            low, high = bootstrap_mean(values, np.random.default_rng(
                SEED + 50000 + 100 * target_index + band_index
            ))
            summary_rows.append({
                "target": target.upper(), "fitzpatrick_band": band,
                "participants": len(values), "participant_MAE": values.mean(),
                "CI_low": low, "CI_high": high,
                "status": "exploratory/underpowered" if band.endswith("V-VI") else "prespecified",
            })
        reference = groups[bands[0]]
        for contrast_index, band in enumerate(bands[1:]):
            comparator = groups[band]
            rng = np.random.default_rng(SEED + 51000 + 100 * target_index + contrast_index)
            draws = (
                comparator[rng.integers(0, len(comparator), size=(BOOTSTRAPS, len(comparator)))].mean(axis=1)
                - reference[rng.integers(0, len(reference), size=(BOOTSTRAPS, len(reference)))].mean(axis=1)
            )
            contrast_rows.append({
                "target": target.upper(),
                "contrast": band + " minus Fitzpatrick I-II",
                "MAE_difference": comparator.mean() - reference.mean(),
                "CI_low": np.quantile(draws, 0.025),
                "CI_high": np.quantile(draws, 0.975),
                "interpretation": (
                    "exploratory/underpowered (n=6 in V-VI)"
                    if band.endswith("V-VI") else "prespecified contrast"
                ),
            })
        trend_data = participant[["fitzpatrick_type", metric]].dropna()
        rho = float(spearmanr(trend_data.fitzpatrick_type, trend_data[metric]).statistic)
        rng = np.random.default_rng(SEED + 52000 + target_index)
        boot = np.empty(BOOTSTRAPS)
        for index in range(BOOTSTRAPS):
            sampled = trend_data.iloc[rng.integers(0, len(trend_data), len(trend_data))]
            boot[index] = spearmanr(sampled.fitzpatrick_type, sampled[metric]).statistic
        permuted = np.empty(PERMUTATIONS)
        x = trend_data.fitzpatrick_type.to_numpy(float)
        y = trend_data[metric].to_numpy(float)
        for index in range(PERMUTATIONS):
            permuted[index] = spearmanr(rng.permutation(x), y).statistic
        trend_rows.append({
            "target": target.upper(), "participants": len(trend_data),
            "spearman_rho": rho,
            "CI_low": np.nanquantile(boot, 0.025),
            "CI_high": np.nanquantile(boot, 0.975),
            "permutation_p_two_sided": (1 + np.sum(np.abs(permuted) >= abs(rho))) / (PERMUTATIONS + 1),
            "interpretation": "exploratory ordinal association; not a fairness test",
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), pd.DataFrame(trend_rows)


def plot_results(summary):
    colors = ["#4C78A8", "#F58518", "#B54A62"]
    labels = ["I–II", "III–IV", "V–VI"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for axis, target, panel in zip(axes, ["SBP", "DBP"], ["A", "B"]):
        shown = summary[summary.target == target].reset_index(drop=True)
        x = np.arange(3)
        y = shown.participant_MAE.to_numpy(float)
        error = np.vstack([y - shown.CI_low.to_numpy(float), shown.CI_high.to_numpy(float) - y])
        axis.errorbar(x, y, yerr=error, fmt="none", ecolor=colors, elinewidth=2, capsize=5)
        axis.scatter(x, y, s=75, c=colors, edgecolor="black", linewidth=0.6, zorder=3)
        axis.set_xticks(x, [f"{label}\n(n={n})" for label, n in zip(labels, shown.participants)])
        axis.set_ylabel("Participant-level MAE (mmHg)")
        axis.set_xlabel("Fitzpatrick band")
        axis.set_title(f"({panel}) {target} MAE")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen tone-blind model error by Fitzpatrick band", fontsize=13)
    fig.savefig(OUTPUT / "fixed_toneblind_mae_by_fitzpatrick.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUTPUT / "fixed_toneblind_mae_by_fitzpatrick.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features = load_or_extract_features()
    print("Feature rows/participants:", len(features), features.pid.nunique(), flush=True)
    ordered = build_replicate_calibrated(features)
    session, session_context = prepare_anchor(ordered, "session_start")
    prior, prior_context = prepare_anchor(ordered, "most_recent_prior")
    uci_models, uci_feature_sets = fit_uci_priors()
    session = add_uci_prior(session, uci_models, uci_feature_sets)
    prior = add_uci_prior(prior, uci_models, uci_feature_sets)
    context_features = [feature for feature in prior_context if feature in set(session_context)]
    people = prior.groupby("subject_id", as_index=False).agg(fst_band=("fst_band", "first"))
    development_people, holdout_people = train_test_split(
        people, test_size=0.25, random_state=SEED + 11100, stratify=people.fst_band
    )
    holdout_ids = set(holdout_people.subject_id.astype(str))
    frames = {}
    for name, frame in [("session", session), ("prior", prior)]:
        frames[name] = {
            "dev": frame[~frame.subject_id.isin(holdout_ids)].copy(),
            "hold": frame[frame.subject_id.isin(holdout_ids)].copy(),
        }
    counts = frames["prior"]["hold"][["subject_id", "fst_band"]].drop_duplicates().fst_band.value_counts()
    expected_counts = {"Fitzpatrick I-II": 133, "Fitzpatrick III-IV": 29, "Fitzpatrick V-VI": 6}
    if counts.to_dict() != expected_counts:
        raise RuntimeError(f"Historical split mismatch: {counts.to_dict()} != {expected_counts}")
    participant_output = None
    selected_rows = []
    for target in ["sbp", "dbp"]:
        base = "baseline_" + target
        candidate_pool = DELTA_RAW + RELATIVE_RAW + [
            "elapsed_seconds", "log_elapsed_days", "session_elapsed_seconds",
            "log_session_elapsed_days", "uci_prior_delta_" + target,
        ]
        shared_pool = [
            feature for feature in dict.fromkeys(candidate_pool)
            if feature in frames["session"]["dev"] and feature in frames["prior"]["dev"]
            and frames["session"]["dev"][feature].notna().any()
            and frames["prior"]["dev"][feature].notna().any()
            and frames["session"]["dev"][feature].nunique(dropna=True) > 1
            and frames["prior"]["dev"][feature].nunique(dropna=True) > 1
        ]
        session_rank = rank_features(frames["session"]["dev"], shared_pool, "delta_" + target).rename(
            columns={"score": "session_score"}
        )
        prior_rank = rank_features(frames["prior"]["dev"], shared_pool, "delta_" + target).rename(
            columns={"score": "prior_score"}
        )
        shared_rank = session_rank.merge(prior_rank, on="feature", how="outer").fillna(0)
        shared_rank["score"] = (shared_rank.session_score + shared_rank.prior_score) / 2
        core = list(dict.fromkeys(
            [base]
            + shared_rank.sort_values(["score", "feature"], ascending=[False, True])
            .feature.head(V12_TOP_K).tolist()
        ))
        model_features = core + context_features
        hold = frames["prior"]["hold"].copy()
        prediction, used, gate = fit_toneblind(
            frames["prior"]["dev"], hold, target, model_features
        )
        hold["prediction"] = prediction
        hold["absolute_error"] = np.abs(prediction - hold[target])
        per = hold.groupby("subject_id", as_index=False).agg(
            fitzpatrick_type=("fitzpatrick_scale", "first"),
            fitzpatrick_band=("fst_band", "first"),
            participant_mae=("absolute_error", "mean"),
            measurements=("absolute_error", "size"),
        )
        per = per.rename(columns={
            "participant_mae": target + "_participant_mae",
            "measurements": target + "_measurements",
        })
        if participant_output is None:
            participant_output = per
        else:
            participant_output = participant_output.merge(
                per, on=["subject_id", "fitzpatrick_type", "fitzpatrick_band"],
                validate="one_to_one",
            )
        selected_rows.append({"target": target, "features": "|".join(used), "gate": repr(gate)})

    expected_band_mae = {
        ("sbp", "Fitzpatrick I-II"): 5.851,
        ("sbp", "Fitzpatrick III-IV"): 5.641,
        ("sbp", "Fitzpatrick V-VI"): 5.295,
        ("dbp", "Fitzpatrick I-II"): 4.450,
        ("dbp", "Fitzpatrick III-IV"): 4.748,
        ("dbp", "Fitzpatrick V-VI"): 4.027,
    }
    observed = {}
    for target in ["sbp", "dbp"]:
        observed.update({
            (target, band): value
            for band, value in participant_output.groupby("fitzpatrick_band")[
                target + "_participant_mae"
            ].mean().items()
        })
    mismatches = {
        str(key): {"observed": observed[key], "expected": expected}
        for key, expected in expected_band_mae.items()
        if abs(observed[key] - expected) > 0.005
    }
    audit = {
        "tone_blind_only": True,
        "skin_features_used": False,
        "skin_rebalancing_used": False,
        "group_specific_models_used": False,
        "holdout_counts": counts.to_dict(),
        "executed_notebook_reference_mae": {str(key): value for key, value in expected_band_mae.items()},
        "reproduction_mismatches_over_0.005_mmhg": mismatches,
        "note": "No result is accepted unless the frozen-model reproduction matches V32.",
    }
    (OUTPUT / "reproduction_audit.json").write_text(json.dumps(audit, indent=2))
    pd.DataFrame(selected_rows).to_csv(OUTPUT / "selected_features_and_gates.csv", index=False)
    if mismatches:
        participant_output.to_csv(OUTPUT / "UNVERIFIED_participant_errors.csv", index=False)
        raise RuntimeError("Exact V32 reproduction failed; see reproduction_audit.json")

    summary, contrasts, trends = analyze_frozen_predictions(participant_output)
    participant_output.to_csv(OUTPUT / "toneblind_participant_errors.csv", index=False)
    summary.to_csv(OUTPUT / "toneblind_band_mae.csv", index=False)
    contrasts.to_csv(OUTPUT / "toneblind_band_contrasts.csv", index=False)
    trends.to_csv(OUTPUT / "toneblind_ordinal_trends.csv", index=False)
    plot_results(summary)
    print("\nBAND MAE\n", summary.round(3).to_string(index=False))
    print("\nCONTRASTS\n", contrasts.round(3).to_string(index=False))
    print("\nEXPLORATORY TRENDS\n", trends.round(4).to_string(index=False))
    print("\nSaved verified outputs to", OUTPUT)


if __name__ == "__main__":
    main()
