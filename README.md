# QRT_energy

Simple experimentation code for training energy price/target prediction models on a DE/FR dataset using:
- XGBoost (`supp_material/xgboost_model.py`)
- TabPFN (`supp_material/TabPFN.py`)

### We reccomend first running the jupiter notebook `supp_material/EDA_and_baseline_models.ipynb` to explore the data and get a feel for baseline models.
Read below only if you run into problems or want to understand the code structure.

Preprocessing lives in `supp_material/preprocessing.py`. CSV saving helper is in `supp_material/utils.py`.

## Expected data files

Scripts assume the following files exist in the *working directory* you run them from:

- `X_train.csv` (features; must include `ID`, `DAY_ID`, `COUNTRY` and various feature columns)
- `y_train.csv` (targets; must include column `TARGET`)
- `X_test.csv` (features; same schema as `X_train.csv` except target)

`utils.save_results_to_csv` will output a CSV with columns:
- `ID`
- `TARGET` (predictions)

## Setup

Create and activate an environment, then install dependencies:

```bash
pip install -U pandas numpy scipy scikit-learn xgboost tabpfn
```

If `tabpfn` installation is problematic on your platform, run only the XGBoost pipeline.

## Preprocessing

Main entrypoint:
- `supp_material.preprocessing.data_preprocessing(X_train, Y_train, X_test, convert_categorical=True)`

What it currently does (high-level):
- builds `is_market_decupled` using a 95th percentile threshold on exchange features
- adds total energy features and leftovers
- adds a `HOLIDAY` boolean based on low global consumption days
- aggregates some fossil energy columns
- drops ID-like columns (`ID`, `DAY_ID`, exchanges)
- optionally converts categorical columns (e.g., `COUNTRY`) to pandas `category`

Adjust preprocessing logic in `supp_material/preprocessing.py`.

## Run: XGBoost

From the repo root:

```bash
python supp_material/xgboost_model.py
```

Notes:
- Hyperparameter search is configured in the `__main__` block via `param_grid`.
- Model uses `enable_categorical=True` (so `COUNTRY` can be categorical).
- The script currently saves predictions via `DataFrame.to_csv` in the `__main__` block; you may prefer using `utils.save_results_to_csv` for consistent formatting.

## Run: TabPFN

From the repo root:

```bash
python supp_material/TabPFN.py
```

What it does:
- preprocesses train/test
- runs a quick 80/20 holdout evaluation (Spearman on the 20%)
- trains on full train and predicts on `X_test`
- saves predictions using `utils.save_results_to_csv`

## Metric

Primary metric used in experiments: **Spearman correlation** (often reported as a raw correlation or multiplied by 100, depending on the function).

## Project structure (relevant)

- `supp_material/preprocessing.py` — feature engineering / cleanup
- `supp_material/utils.py` — CSV saving helper (`save_results_to_csv`)
- `supp_material/xgboost_model.py` — XGBoost training + CV + evaluation utilities
- `supp_material/TabPFN.py` — TabPFN training + holdout eval + predictions
````

</file>
