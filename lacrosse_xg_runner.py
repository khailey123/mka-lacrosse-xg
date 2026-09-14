#!/usr/bin/env python3
"""
MKA Lacrosse xG runner — standalone JSON interface.
Reads JSON from stdin, runs RF / XGBoost with leave-one-game-out CV, writes JSON to stdout.
Also computes Logistic Regression baseline for comparison.
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import shap

# ── Feature schema ────────────────────────────────────────────────────────────
CAT_COLS = ["zone", "coverage", "situation", "hand", "height", "shotFlight"]
NUM_COLS = ["offPass"]

# Feature column sets by mode (season-analysis only):
#   'all'  — full feature set (default, current behaviour)
#   'pre'  — pre-shot features: zone + coverage + situation + hand + offPass
#   'post' — post-shot features: zone + height + shotFlight
FEATURE_COLS_BY_MODE = {
    'all':  (["zone", "coverage", "situation", "hand", "height", "shotFlight"], ["offPass"]),
    'pre':  (["zone", "coverage", "situation"], ["offPass"]),
    'post': (["zone", "hand", "height", "shotFlight"], []),
}

# "situation" categories are sport-specific: girls lacrosse uses "penalty_shot"
# (8 Meter), boys lacrosse uses "odd_man_situation" (formerly "man_up"/"man_down",
# which are normalized to "odd_man_situation" at read time in situation.ts).
SITUATION_CATS_BY_SPORT = {
    "girls_lacrosse": ["transition", "settled", "penalty_shot"],
    "boys_lacrosse":  ["transition", "settled", "odd_man_situation"],
}

def known_cats_for_sport(sport):
    return {
        "zone":       [str(z) for z in range(1, 20)],
        "coverage":   ["wide_open", "lightly_covered", "heavily_covered"],
        "situation":  SITUATION_CATS_BY_SPORT.get(sport, SITUATION_CATS_BY_SPORT["girls_lacrosse"]),
        "hand":       ["left", "right"],
        "height":     ["low", "high"],
        "shotFlight": ["air", "bouncer"],
    }

# ── Preprocessor ──────────────────────────────────────────────────────────────
def build_preprocessor(known_cats, cat_cols=None, num_cols=None):
    if cat_cols is None: cat_cols = CAT_COLS
    if num_cols is None: num_cols = NUM_COLS
    transformers = []
    for col in cat_cols:
        enc = OneHotEncoder(
            categories=[known_cats[col]],
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )
        transformers.append((f"cat_{col}", enc, [col]))
    if num_cols:
        from sklearn.preprocessing import FunctionTransformer
        transformers.append(("num", FunctionTransformer(), num_cols))
    return ColumnTransformer(transformers)

def get_feature_names(preprocessor, known_cats):
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name.startswith("cat_"):
            col = cols[0]
            categories = known_cats[col][1:]  # drop='first' removes first category
            names.extend([f"{col}={c}" for c in categories])
        else:
            names.extend(cols)
    return names

# ── Data preparation ──────────────────────────────────────────────────────────
def shots_to_df(shots, known_cats, cat_cols=None, num_cols=None):
    if cat_cols is None: cat_cols = CAT_COLS
    if num_cols is None: num_cols = NUM_COLS
    if not shots:
        raise ValueError("No shots provided — cannot train model on empty dataset.")
    df = pd.DataFrame(shots)
    df["zone"] = df["zone"].astype(str)
    if "offPass" in num_cols:
        df["offPass"] = df.get("offPass", 0).astype(float).fillna(0.0)
    df["is_goal"] = (df["outcome"] == "goal").astype(int)
    for col in cat_cols:
        if col not in df.columns:
            df[col] = known_cats[col][0]
        df[col] = df[col].fillna(known_cats[col][0]).astype(str)
        if col == "situation":
            # Guard against any stray cross-sport situation value that
            # shouldn't exist for this sport's data (defensive; data is
            # already sport-scoped upstream in the API server).
            valid = set(known_cats[col])
            df[col] = df[col].where(df[col].isin(valid), known_cats[col][0])
    return df

# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(model_type):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    elif model_type == "logistic":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ── LOGO-CV core ──────────────────────────────────────────────────────────────
def run_logo_cv(df, preprocessor, feature_names, model_type, folds, cat_cols=None, num_cols=None):
    """Run LOGO-CV for given folds. Returns dicts of predictions, actuals, shap, shot-level."""
    if cat_cols is None: cat_cols = CAT_COLS
    if num_cols is None: num_cols = NUM_COLS
    feature_cols = cat_cols + num_cols

    all_probs       = []
    all_actuals     = []
    all_game_ids    = []
    all_shap_values = []
    all_X_test      = []   # parallel list to all_shap_values
    shot_preds      = []

    for held_gid in folds:
        train_df = df[df["gameId"] != held_gid]
        test_df  = df[df["gameId"] == held_gid]

        if len(train_df) < 10 or len(test_df) == 0:
            continue
        if len(set(train_df["is_goal"])) < 2:
            continue

        X_train = preprocessor.transform(train_df[feature_cols])
        y_train = train_df["is_goal"].values
        X_test  = preprocessor.transform(test_df[feature_cols])
        y_test  = test_df["is_goal"].values

        clf = build_model(model_type)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        # SHAP (tree models only)
        if model_type in ("random_forest", "xgboost"):
            try:
                explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
                sv = explainer.shap_values(X_test)
                # Handle all SHAP output formats:
                # old (<0.40): list [neg_class_arr, pos_class_arr]
                # new (>=0.40): ndarray (n_samples, n_features) for XGB
                #               ndarray (n_samples, n_features, n_classes) for RF
                if isinstance(sv, list):
                    sv = np.array(sv[1])       # old format — positive class
                else:
                    sv = np.array(sv)
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]       # new 3-D RF format — positive class slice
                # sv is now guaranteed (n_samples, n_features)
                all_shap_values.append(sv)
                all_X_test.append(X_test)
            except Exception:
                pass

        all_probs.extend(probs.tolist())
        all_actuals.extend(y_test.tolist())
        all_game_ids.extend([held_gid] * len(test_df))

        for i, (prob, actual) in enumerate(zip(probs.tolist(), y_test.tolist())):
            row = test_df.iloc[i]
            zone_raw = row.get("zone", "0")
            try:
                zone_int = int(zone_raw)
            except (ValueError, TypeError):
                zone_int = 0
            shot_preds.append({
                "gameId":      int(held_gid) if held_gid is not None else None,
                "zone":        zone_int,
                "outcome":     str(row.get("outcome", "")),
                "coverage":    str(row.get("coverage", "")),
                "situation":   str(row.get("situation", "")),
                "hand":        str(row.get("hand", "")),
                "height":      str(row.get("height", "")),
                "shotFlight":  str(row.get("shotFlight", "")),
                "offPass":     bool(row.get("offPass", False)),
                "predictedXG": float(prob),
                "isGoal":      int(actual),
            })

    return {
        "probs":    all_probs,
        "actuals":  all_actuals,
        "gameIds":  all_game_ids,
        "shapVals": all_shap_values,
        "xVals":    all_X_test,
        "shots":    shot_preds,
    }

def compute_calibration(actuals, probs, n_bins=10):
    """Reliability diagram data + Expected Calibration Error (ECE)."""
    if len(set(actuals)) < 2 or len(probs) < 10:
        return None
    try:
        probs_arr   = np.array(probs)
        actuals_arr = np.array(actuals)
        n = len(probs_arr)
        bins = []
        ece  = 0.0
        edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probs_arr >= lo) & (probs_arr < hi)
            if i == n_bins - 1:
                mask = (probs_arr >= lo) & (probs_arr <= hi)
            cnt = int(mask.sum())
            if cnt > 0:
                mean_pred = float(probs_arr[mask].mean())
                frac_pos  = float(actuals_arr[mask].mean())
                bins.append({
                    "meanPred": round(mean_pred, 4),
                    "fracPos":  round(frac_pos,  4),
                    "count":    cnt,
                })
                ece += cnt / n * abs(mean_pred - frac_pos)
        return {"bins": bins, "ece": round(float(ece), 4)}
    except Exception:
        return None

def compute_metrics(actuals, probs):
    if len(set(actuals)) < 2:
        return {}
    try:
        return {
            "logLoss":    round(float(log_loss(actuals, probs)), 4),
            "brierScore": round(float(brier_score_loss(actuals, probs)), 4),
            "rocAuc":     round(float(roc_auc_score(actuals, probs)), 4),
        }
    except Exception:
        return {}

def compute_shap_importance(all_shap_values, all_X_values, feature_names):
    """
    Compute per-feature SHAP importance.

    meanShap uses *conditional* mean SHAP — the average SHAP value across shots
    where that feature is active (X > 0.5).  For one-hot encoded features this
    means "when the shot actually has this property, how much does it push xG?"
    This avoids the confusion of averaging over shots where the feature is 0
    (reference / inactive), which dilutes and often inverts the apparent sign.
    Falls back to global mean when fewer than 5 active shots exist.
    """
    if not all_shap_values:
        return []
    stacked_sv = np.vstack(all_shap_values)   # (N, n_features)
    stacked_X  = np.vstack(all_X_values)      # (N, n_features)
    mean_abs   = np.abs(stacked_sv).mean(axis=0)

    results = []
    for i, f in enumerate(feature_names):
        shap_col = stacked_sv[:, i]
        x_col    = stacked_X[:, i]
        active   = x_col > 0.5            # OHE / binary feature is "on"
        if active.sum() >= 5:
            cond_shap = float(shap_col[active].mean())
        else:
            cond_shap = float(shap_col.mean())   # fallback: global mean
        results.append((f, float(mean_abs[i]), cond_shap))

    results.sort(key=lambda x: -x[1])     # descending by magnitude
    return [
        {"feature": f, "importance": round(abs_v, 5), "meanShap": round(cond_v, 5)}
        for f, abs_v, cond_v in results
    ]

def build_per_game_xg(game_ids, probs, actuals):
    per_game = {}
    for gid, prob, actual in zip(game_ids, probs, actuals):
        key = str(int(gid))
        if key not in per_game:
            per_game[key] = {"xg": 0.0, "shots": 0, "goals": 0}
        per_game[key]["xg"]    = round(per_game[key]["xg"] + prob, 4)
        per_game[key]["shots"] += 1
        if actual == 1:
            per_game[key]["goals"] += 1
    return per_game

# ── Predict single shot (simulator) ───────────────────────────────────────────
def predict_shot_action(payload):
    """Train on ALL data (for this sport), predict one hypothetical shot, return xG + per-shot SHAP."""
    shots        = payload["shots"]
    model_type   = payload.get("model", "random_forest")
    shot         = payload["shot"]
    sport        = payload.get("sport", "girls_lacrosse")
    feature_mode = payload.get("featureMode", "all")
    known_cats   = known_cats_for_sport(sport)
    cat_cols, num_cols = FEATURE_COLS_BY_MODE.get(feature_mode, FEATURE_COLS_BY_MODE["all"])

    df = shots_to_df(shots, known_cats, cat_cols, num_cols)
    if len(df) < 10:
        print(json.dumps({"error": "Not enough shots to train a model."}))
        return
    if len(set(df["is_goal"])) < 2:
        print(json.dumps({"error": "Need both goal and non-goal shots to train."}))
        return

    feature_cols = cat_cols + num_cols
    preprocessor = build_preprocessor(known_cats, cat_cols, num_cols)
    preprocessor.fit(df[feature_cols])
    feature_names = get_feature_names(preprocessor, known_cats)

    X_all = preprocessor.transform(df[feature_cols])
    y_all = df["is_goal"].values

    clf = build_model(model_type)
    clf.fit(X_all, y_all)

    # Build single-shot row — only include columns active for this mode
    shot_all = {
        "zone":       str(shot.get("zone",       "1")),
        "coverage":   shot.get("coverage",        "wide_open"),
        "situation":  shot.get("situation",        "settled"),
        "hand":       shot.get("hand",             "right"),
        "height":     shot.get("height",           "low"),
        "shotFlight": shot.get("shotFlight",       "air"),
        "offPass":    float(bool(shot.get("offPass", False))),
    }
    shot_df = pd.DataFrame([{k: shot_all[k] for k in feature_cols if k in shot_all}])
    X_shot = preprocessor.transform(shot_df[feature_cols])
    prob   = float(clf.predict_proba(X_shot)[0, 1])

    # Per-shot SHAP waterfall
    shap_values = []
    base_value  = None
    try:
        if model_type in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
            sv = explainer.shap_values(X_shot)
            if isinstance(sv, list):
                sv = np.array(sv[1])
            else:
                sv = np.array(sv)
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
            # sv: (1, n_features)
            raw_base = explainer.expected_value
            if isinstance(raw_base, (list, np.ndarray)):
                base_value = float(raw_base[1])
            else:
                base_value = float(raw_base)
            for i, fname in enumerate(feature_names):
                sv_i = float(sv[0, i])
                if abs(sv_i) > 1e-6:
                    shap_values.append({
                        "feature": fname,
                        "shap":    round(sv_i, 5),
                        "value":   round(float(X_shot[0, i]), 3),
                    })
            shap_values.sort(key=lambda x: -abs(x["shap"]))
    except Exception:
        pass

    print(json.dumps({
        "predictedXG": round(prob, 4),
        "shapValues":  shap_values,
        "baseValue":   base_value,
        "model":       model_type,
    }))

# ── Main entry ────────────────────────────────────────────────────────────────
def main():
    payload        = json.loads(sys.stdin.read())

    if payload.get("action") == "predict_shot":
        predict_shot_action(payload)
        return

    shots          = payload["shots"]
    model_type     = payload.get("model", "random_forest")
    held_out_gid   = payload.get("heldOutGameId", None)
    sport          = payload.get("sport", "girls_lacrosse")
    feature_mode   = payload.get("featureMode", "all")
    known_cats     = known_cats_for_sport(sport)
    cat_cols, num_cols = FEATURE_COLS_BY_MODE.get(feature_mode, FEATURE_COLS_BY_MODE["all"])

    df = shots_to_df(shots, known_cats, cat_cols, num_cols)
    game_ids = sorted(df["gameId"].dropna().unique().tolist())

    if len(game_ids) < 2:
        print(json.dumps({"error": "Need at least 2 games for leave-one-game-out cross-validation."}))
        return

    feature_cols = cat_cols + num_cols
    # Fit preprocessor on ALL data once (consistent feature space)
    preprocessor = build_preprocessor(known_cats, cat_cols, num_cols)
    preprocessor.fit(df[feature_cols])
    feature_names = get_feature_names(preprocessor, known_cats)

    folds = [held_out_gid] if held_out_gid is not None else game_ids

    # ── Primary model ─────────────────────────────────────────────────────────
    res = run_logo_cv(df, preprocessor, feature_names, model_type, folds, cat_cols, num_cols)

    per_game_xg     = build_per_game_xg(res["gameIds"], res["probs"], res["actuals"])
    metrics         = compute_metrics(res["actuals"], res["probs"])
    shap_importance = compute_shap_importance(res["shapVals"], res["xVals"], feature_names)
    calibration     = compute_calibration(res["actuals"], res["probs"])

    # ── Logistic Regression baseline (for comparison — season mode only) ──────
    logistic_metrics = {}
    if held_out_gid is None:
        try:
            lr_res = run_logo_cv(df, preprocessor, feature_names, "logistic", game_ids, cat_cols, num_cols)
            logistic_metrics = compute_metrics(lr_res["actuals"], lr_res["probs"])
        except Exception:
            pass

    output = {
        "model":           model_type,
        "perGameXG":       per_game_xg,
        "metrics":         metrics,
        "shapImportance":  shap_importance,
        "shotPredictions": res["shots"],
        "foldCount":       len(set(res["gameIds"])),
        "logisticMetrics": logistic_metrics,
        "calibration":     calibration,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}))
        sys.exit(1)
