#!/usr/bin/env python3
"""Evaluate all public MKA lacrosse xG models with leave-one-game-group-out CV."""

import argparse
import json
from pathlib import Path

import pandas as pd

from lacrosse_xg_runner import (
    FEATURE_COLS_BY_MODE,
    build_preprocessor,
    compute_calibration,
    compute_metrics,
    get_feature_names,
    known_cats_for_sport,
    run_logo_cv,
    shots_to_df,
)

SPORTS = ("girls_lacrosse", "boys_lacrosse")
MODELS = ("logistic", "random_forest", "xgboost")


def load_shots(path: Path, sport: str) -> list[dict]:
    source = pd.read_csv(path)
    source = source[source["sport"] == sport].copy()
    source = source.rename(
        columns={
            "game_group": "gameId",
            "shot_flight": "shotFlight",
            "off_pass": "offPass",
        }
    )
    source["offPass"] = (
        source["offPass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
        .astype(int)
    )
    return source.to_dict(orient="records")


def evaluate(path: Path, sport: str, model: str, feature_mode: str) -> dict:
    known_cats = known_cats_for_sport(sport)
    cat_cols, num_cols = FEATURE_COLS_BY_MODE[feature_mode]
    df = shots_to_df(load_shots(path, sport), known_cats, cat_cols, num_cols)
    groups = sorted(df["gameId"].dropna().unique().tolist())
    if len(groups) < 2:
        raise ValueError(f"{sport} requires at least two anonymized game groups")

    feature_cols = cat_cols + num_cols
    preprocessor = build_preprocessor(known_cats, cat_cols, num_cols)
    preprocessor.fit(df[feature_cols])
    feature_names = get_feature_names(preprocessor, known_cats)
    result = run_logo_cv(
        df,
        preprocessor,
        feature_names,
        model,
        groups,
        cat_cols,
        num_cols,
    )
    return {
        "sport": sport,
        "model": model,
        "featureMode": feature_mode,
        "shots": len(result["probs"]),
        "folds": len(set(result["gameIds"])),
        "metrics": compute_metrics(result["actuals"], result["probs"]),
        "calibration": compute_calibration(result["actuals"], result["probs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("lacrosse_xg_shots_anonymized.csv"),
    )
    parser.add_argument("--sport", choices=(*SPORTS, "all"), default="all")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--feature-mode", choices=("all", "pre", "post"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sports = SPORTS if args.sport == "all" else (args.sport,)
    models = MODELS if args.model == "all" else (args.model,)
    report = [
        evaluate(args.data, sport, model, args.feature_mode)
        for sport in sports
        for model in models
    ]
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()