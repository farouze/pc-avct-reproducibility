def build_delta_table(frame, raw_features, pc_features, anchor='session_start'):
    if anchor not in {'session_start', 'most_recent_prior'}:
        raise ValueError("anchor must be 'session_start' or 'most_recent_prior'")
    features = list(dict.fromkeys(list(raw_features) + list(pc_features)))
    rows = []
    for subject_id, g in frame.groupby('subject_id', sort=False):
        order_columns = [c for c in ['start_seconds', '_source_order'] if c in g.columns]
        g = g.sort_values(order_columns, na_position='last') if order_columns else g.copy()
        if len(g) < 2:
            continue
        session = g.iloc[0]
        for position in range(1, len(g)):
            row = g.iloc[position]
            prior = g.iloc[position - 1]
            reference = prior if anchor == 'most_recent_prior' else session
            item = row.to_dict()
            item['original_index'] = g.index[position]
            item['anchor_mode'] = anchor
            item['anchor_start_seconds'] = pd.to_numeric(
                pd.Series([reference.get('start_seconds')]), errors='coerce').iloc[0]
            item['anchor_phase'] = reference.get('phase', np.nan)
            item['anchor_measurement'] = reference.get('measurement', np.nan)

            row_time = pd.to_numeric(pd.Series([row.get('start_seconds')]), errors='coerce').iloc[0]
            ref_time = pd.to_numeric(pd.Series([reference.get('start_seconds')]), errors='coerce').iloc[0]
            session_time = pd.to_numeric(pd.Series([session.get('start_seconds')]), errors='coerce').iloc[0]
            item['elapsed_seconds'] = float(row_time - ref_time) if np.isfinite(row_time) and np.isfinite(ref_time) else np.nan
            item['prior_gap_seconds'] = item['elapsed_seconds']
            item['session_elapsed_seconds'] = float(row_time - session_time) if np.isfinite(row_time) and np.isfinite(session_time) else np.nan

            for target in ['sbp', 'dbp']:
                current = pd.to_numeric(pd.Series([row.get(target)]), errors='coerce').iloc[0]
                ref_value = pd.to_numeric(pd.Series([reference.get(target)]), errors='coerce').iloc[0]
                session_value = pd.to_numeric(pd.Series([session.get(target)]), errors='coerce').iloc[0]
                item['baseline_' + target] = float(ref_value) if np.isfinite(ref_value) else np.nan
                item['delta_' + target] = float(current - ref_value) if np.isfinite(current) and np.isfinite(ref_value) else np.nan
                item['session_baseline_' + target] = float(session_value) if np.isfinite(session_value) else np.nan
                item['session_delta_' + target] = float(current - session_value) if np.isfinite(current) and np.isfinite(session_value) else np.nan

            for feature in features:
                current = pd.to_numeric(pd.Series([row.get(feature)]), errors='coerce').iloc[0]
                initial = pd.to_numeric(pd.Series([reference.get(feature)]), errors='coerce').iloc[0]
                valid = np.isfinite(current) and np.isfinite(initial)
                item['delta_' + feature] = float(current - initial) if valid else np.nan
                denominator = max(abs(initial), 1e-6) if np.isfinite(initial) else np.nan
                relative = (current - initial) / denominator if valid else np.nan
                item['relative_' + feature] = float(np.clip(relative, -10, 10)) if np.isfinite(relative) else np.nan
            rows.append(item)
    return pd.DataFrame(rows)

def add_aurora_measurement_context(frame):
    """Create only context features supported by columns/labels in `frame`."""
    out = frame.copy()
    created, audit = [], []

    if 'phase' in out.columns:
        phase = out.phase.fillna('missing').astype(str).str.strip().str.lower()
        for value in sorted(v for v in phase.unique() if v and v != 'missing'):
            safe = re.sub(r'[^a-z0-9]+', '_', value).strip('_')
            column = 'context_phase_' + safe
            out[column] = (phase == value).astype(float)
            created.append(column)
        if phase.eq('ambulatory').any():
            out['context_is_ambulatory'] = phase.eq('ambulatory').astype(float)
            out['context_is_in_lab'] = phase.isin(['initial', 'return']).astype(float)
            created += ['context_is_ambulatory', 'context_is_in_lab']
            audit.append({'source': 'phase', 'derived': 'in-lab/ambulatory', 'status': 'added'})
        else:
            audit.append({'source': 'phase', 'derived': 'in-lab/ambulatory',
                          'status': 'not added: no ambulatory rows in this subset'})

    label = (out['measurement'].fillna('').astype(str).str.lower()
             if 'measurement' in out.columns else pd.Series('', index=out.index))
    semantic_patterns = {
        'context_posture_supine': r'\bsupine\b',
        'context_posture_seated': r'\b(seated|sitting)\b',
        'context_posture_standing': r'\bstanding\b',
        'context_activity_post_exercise': r'\b(post[ -]?exercise|exercise)\b',
        'context_measurement_calibration': r'\bcalibration\b',
    }
    for column, pattern in semantic_patterns.items():
        matched = label.str.contains(pattern, regex=True, na=False)
        if matched.any():
            out[column] = matched.astype(float)
            created.append(column)
            audit.append({'source': 'measurement label', 'derived': column,
                          'status': f'added ({int(matched.sum())} rows)'})
        else:
            audit.append({'source': 'measurement label', 'derived': column,
                          'status': 'not added: no matching observed labels'})

    numeric_context = {
        'duration': 'context_duration_seconds',
        'pressure_quality': 'context_pressure_quality',
        'optical_quality': 'context_optical_quality',
    }
    for source, column in numeric_context.items():
        if source in out.columns:
            values = pd.to_numeric(out[source], errors='coerce')
            if values.notna().any():
                out[column] = values
                created.append(column)
                audit.append({'source': source, 'derived': column, 'status': 'added'})
            else:
                audit.append({'source': source, 'derived': column, 'status': 'not added: all missing'})

    created = list(dict.fromkeys(created))
    return out, created, pd.DataFrame(audit)

def normalized_conditional_mi(frame,features,target,conditioning):
    conditioning=[c for c in conditioning if c in frame.columns]
    C=SimpleImputer(strategy='median').fit_transform(frame[conditioning])
    C=RobustScaler().fit_transform(C)
    y=np.asarray(frame[target],float)
    y_res=y-Ridge(alpha=1).fit(C,y).predict(C)
    X=SimpleImputer(strategy='median').fit_transform(frame[features])
    X_res=np.empty_like(X,float)
    for j in range(X.shape[1]):
        X_res[:,j]=X[:,j]-Ridge(alpha=1).fit(C,X[:,j]).predict(C)
    values=mutual_info_regression(X_res,y_res,random_state=SEED)
    return values/max(float(np.max(values)),1e-12)

def make_model_v17(params, loss='squared_error'):
    """make_model, but with the loss exposed. The default is squared_error."""
    return make_pipeline(SimpleImputer(strategy='median'),
                         HistGradientBoostingRegressor(max_iter=300, loss=loss,
                                                       random_state=SEED, **params))

def abl_weights(frame, band_col, use_skin):
    visits = frame.groupby('subject_id').subject_id.transform('size').astype(float)
    w = 1.0 / visits
    if use_skin:
        people = frame[['subject_id', band_col]].drop_duplicates()
        counts = people[band_col].value_counts()
        w = w * frame[band_col].map(
            (len(people) / (len(counts) * counts)).to_dict()).astype(float)
    w = np.clip(w, np.quantile(w, .02), np.quantile(w, .98))
    return np.asarray(w / w.mean(), float)

def abl_rank(frame, features, target, band_col, use_group):
    features = [f for f in dict.fromkeys(features)
                if f in frame and frame[f].notna().any()]
    pressure = 'sbp' if target.endswith('sbp') else 'dbp'
    conditioning = ['baseline_' + pressure, 'log_elapsed_days']
    global_mi = normalized_conditional_mi(frame, features, target, conditioning)
    if not use_group:
        return (pd.DataFrame({'feature': features, 'score': global_mi})
                  .sort_values('score', ascending=False).reset_index(drop=True))
    group_mi = []
    for _, g in frame.groupby(band_col):
        if len(g) >= 30 and g[target].nunique() > 3:
            group_mi.append(normalized_conditional_mi(g, features, target, conditioning))
    worst = np.min(np.vstack(group_mi), axis=0) if group_mi else global_mi
    return (pd.DataFrame({'feature': features, 'score': .5 * global_mi + .5 * worst})
              .sort_values('score', ascending=False).reset_index(drop=True))

def abl_gate_search(g, target, base, dp):
    baseline_mae = float(g.assign(_ae=np.abs(g[base] - g[target]))
                          .groupby('subject_id')['_ae'].mean().mean())
    best = None
    for threshold in V10_THRESHOLD_GRID:
        active = np.abs(dp) >= threshold
        for alpha in V10_ALPHA_GRID:
            pred = g[base].to_numpy(float) + np.where(active, alpha * dp, 0.0)
            mae = float(g.assign(_ae=np.abs(pred - g[target]))
                         .groupby('subject_id')['_ae'].mean().mean())
            if mae <= baseline_mae + 1e-12:
                if best is None or (mae, float(threshold)) < (best['mae'], best['threshold']):
                    best = {'threshold': float(threshold), 'alpha': float(alpha), 'mae': mae}
    return best or {'threshold': float('inf'), 'alpha': 0.0, 'mae': baseline_mae}

def abl_fit_gate(frame, target, base, dp, band_col, per_group):
    working = frame.assign(_dp=np.asarray(dp, float))
    if not per_group:
        return {'_global': abl_gate_search(working, target, base,
                                           working._dp.to_numpy(float))}
    return {group: abl_gate_search(g, target, base, g._dp.to_numpy(float))
            for group, g in working.groupby(band_col)}

def abl_apply_gate(frame, base, dp, gate, band_col):
    dp = np.asarray(dp, float)
    if '_global' in gate:
        threshold = np.full(len(frame), gate['_global']['threshold'])
        alpha = np.full(len(frame), gate['_global']['alpha'])
    else:
        threshold = frame[band_col].map(
            {k: v['threshold'] for k, v in gate.items()}).fillna(np.inf).to_numpy(float)
        alpha = frame[band_col].map(
            {k: v['alpha'] for k, v in gate.items()}).fillna(0.0).to_numpy(float)
    active = np.abs(dp) >= threshold
    return frame[base].to_numpy(float) + np.where(active, alpha * dp, 0.0), active

def v26_build_calibration_dose(frame, anchor_readings):
    # Build a common post-calibration cohort using exactly k anchor readings.
    # Scoring begins only after the complete initial calibration block, so all dose arms
    # see identical future rows. Later calibration-labelled rows remain auditable.
    source = frame.copy()
    source["sbp"] = pd.to_numeric(source["sbp"], errors="coerce")
    source["dbp"] = pd.to_numeric(source["dbp"], errors="coerce")
    source["_parsed_time"] = pd.to_datetime(source["date_time"], errors="coerce")
    source["_is_initial_calibration"] = (
        source["phase"].astype(str).str.lower().eq("initial")
        & source["measurement"].astype(str).str.contains(
            V26_CAL_PATTERN, case=False, regex=True, na=False
        )
    )
    numeric = source.select_dtypes(include=[np.number]).columns.tolist()
    waveform_features = [
        column for column in numeric if column.startswith(("raw_", "pc_"))
    ]
    rows, audit = [], []
    for pid, group in source.dropna(subset=["sbp", "dbp"]).groupby("pid"):
        group = group.sort_values(
            ["_parsed_time", "_source_order"], na_position="last"
        )
        calibrations = group[group["_is_initial_calibration"]].copy()
        if len(calibrations) < anchor_readings:
            audit.append({
                "pid": pid, "status": "excluded_insufficient_calibration",
                "available": len(calibrations), "anchor_readings": anchor_readings,
            })
            continue
        selected = calibrations.head(anchor_readings)
        calibration_end_order = calibrations["_source_order"].max()
        calibration_end_time = calibrations["_parsed_time"].max()

        anchor = selected.iloc[0].copy()
        anchor["sbp"] = selected["sbp"].mean()
        anchor["dbp"] = selected["dbp"].mean()
        for column in waveform_features:
            anchor[column] = pd.to_numeric(
                selected[column], errors="coerce"
            ).median()
        # Time zero is the end of the full calibration block for every dose arm.
        anchor["_source_order"] = calibration_end_order
        anchor["_parsed_time"] = calibration_end_time
        anchor["date_time"] = calibration_end_time
        anchor["measurement"] = f"V26 anchor from {anchor_readings} reading(s)"
        anchor["subject_id"] = str(pid)
        anchor["start_seconds"] = 0.0

        if pd.notna(calibration_end_time):
            future = group[
                (~group["_is_initial_calibration"])
                & (
                    (group["_parsed_time"] > calibration_end_time)
                    | (
                        group["_parsed_time"].eq(calibration_end_time)
                        & group["_source_order"].gt(calibration_end_order)
                    )
                )
            ]
        else:
            future = group[
                (~group["_is_initial_calibration"])
                & group["_source_order"].gt(calibration_end_order)
            ]
        if future.empty:
            audit.append({
                "pid": pid, "status": "excluded_no_future",
                "available": len(calibrations), "anchor_readings": anchor_readings,
            })
            continue

        rows.append(anchor)
        for _, current in future.iterrows():
            current = current.copy()
            current["subject_id"] = str(pid)
            if pd.notna(current["_parsed_time"]) and pd.notna(calibration_end_time):
                current["start_seconds"] = (
                    current["_parsed_time"] - calibration_end_time
                ).total_seconds()
            else:
                current["start_seconds"] = float(
                    (current["_source_order"] - calibration_end_order) * WINDOW_SECONDS
                )
            rows.append(current)
        audit.append({
            "pid": pid, "status": "included", "available": len(calibrations),
            "anchor_readings": anchor_readings, "future_rows": len(future),
        })
    return pd.DataFrame(rows), pd.DataFrame(audit)

def v26_prepare_dose(anchor_readings):
    ordered, audit = v26_build_calibration_dose(
        aurora_features, anchor_readings
    )
    frame = build_delta_table(
        ordered, RAW, PC, anchor="session_start"
    ).replace([np.inf, -np.inf], np.nan)
    frame, context, context_audit = add_aurora_measurement_context(frame)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["fst_band"] = pd.cut(
        frame["fitzpatrick_scale"], [0, 2, 4, 6],
        labels=["Fitzpatrick I-II", "Fitzpatrick III-IV", "Fitzpatrick V-VI"],
    ).astype(str)
    frame["log_elapsed_days"] = np.log1p(
        frame["elapsed_seconds"].clip(lower=0) / 86400
    )
    frame["log_session_elapsed_days"] = np.log1p(
        frame["session_elapsed_seconds"].clip(lower=0) / 86400
    )
    frame["v26_is_calibration_labeled"] = frame["measurement"].astype(str).str.contains(
        V26_ANY_CAL_PATTERN, case=False, regex=True, na=False
    )
    for target in ["sbp", "dbp"]:
        prior_features = uci_prior_features[target]
        if all(feature in frame.columns for feature in prior_features):
            frame["uci_prior_delta_" + target] = uci_prior_models[target].predict(
                frame[prior_features]
            )
    return frame, context, audit, context_audit

def v26_row_key(frame):
    return pd.MultiIndex.from_frame(
        # _source_order comes from the original Aurora table and is stable even when a
        # stricter calibration dose excludes additional participants before reconstruction.
        frame[["subject_id", "_source_order"]].astype(str)
    )

def v26_participant_folds(frame, n_splits, repeat):
    people = frame[["subject_id", "fst_band"]].drop_duplicates("subject_id")
    people = people.sort_values("subject_id").reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=SEED + 26200 + repeat,
    )
    for fold, (train_people, test_people) in enumerate(
        splitter.split(people["subject_id"], people["fst_band"]), 1
    ):
        train_ids = set(people.iloc[train_people]["subject_id"])
        test_ids = set(people.iloc[test_people]["subject_id"])
        train = frame[frame["subject_id"].isin(train_ids)].copy()
        test = frame[frame["subject_id"].isin(test_ids)].copy()
        assert not set(train["subject_id"]) & set(test["subject_id"])
        yield fold, train, test

def v26_core_features(train, target):
    base = "baseline_" + target
    pool = [
        feature for feature in dict.fromkeys(
            DELTA_RAW + RELATIVE_RAW + [
                "elapsed_seconds", "log_elapsed_days",
                "session_elapsed_seconds", "log_session_elapsed_days",
                "uci_prior_delta_" + target,
            ]
        )
        if feature in train.columns and train[feature].notna().any()
        and train[feature].nunique(dropna=True) > 1
    ]
    ranking = abl_rank(
        train.dropna(subset=["delta_" + target]), pool,
        "delta_" + target, "fst_band", False,
    )
    return list(dict.fromkeys(
        [base] + ranking["feature"].head(V26_TOP_K).tolist()
    ))
