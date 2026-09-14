# MKA Lacrosse xG Open-Source Preparation

This directory is a self-contained public release of the boys and girls lacrosse expected-goals models.

## Model families

Both sports support multiple model families:

- Logistic regression
- Random forest
- XGBoost

The models use sport-aware feature categories so boys and girls lacrosse can be trained and evaluated separately.

Each model can be evaluated with the full, pre-shot, or post-shot feature set. Evaluation uses leave-one-anonymized-game-group-out cross-validation and reports log loss, Brier score, ROC AUC, and calibration.

## Included material

- `lacrosse_xg_runner.py`: training, prediction, LOGO-CV, calibration, and SHAP logic
- `evaluate_models.py`: command-line evaluation for both sports and all models
- `lacrosse_xg_shots_anonymized.csv`: public-safe model dataset
- `requirements.txt`: pinned Python dependencies
- `LICENSE`: MIT license

## Run

```bash
python -m pip install -r requirements.txt
python evaluate_models.py
```

Evaluate one combination:

```bash
python evaluate_models.py --sport boys_lacrosse --model xgboost
```

Available sports are `boys_lacrosse` and `girls_lacrosse`. Available models are `logistic`, `random_forest`, and `xgboost`.

## Public dataset

`lacrosse_xg_shots_anonymized.csv` contains only:

- `sport`
- `game_group`
- `team`
- `zone`
- `coverage`
- `situation`
- `hand`
- `height`
- `shot_flight`
- `off_pass`
- `outcome`

`game_group` is a sequential identifier created independently within each sport. It preserves game-level cross-validation without exposing the original database game ID.

The public dataset excludes player numbers, player names, original game IDs, dates, opponents, locations, scores, and other game-identifying metadata.

## Private source data

The original `lacrosse_data_export.csv` is intentionally excluded from Git. It contains jersey numbers and game-level quasi-identifiers and must not be included in a public repository.

## Environment configuration

Copy `.env.example` to `.env` for local development and provide real values locally. Never commit `.env`.