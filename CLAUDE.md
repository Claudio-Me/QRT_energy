# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Energy price prediction models for DE/FR dataset using XGBoost and TabPFN. The primary metric is **Spearman correlation** (often scaled by 100).

## Dependencies

```bash
pip install -U pandas numpy scipy scikit-learn xgboost tabpfn
```

If `tabpfn` installation fails, run only XGBoost pipelines.

## Data Files

Scripts expect these files in the working directory:
- `X_train.csv` - Training features (includes `ID`, `DAY_ID`, `COUNTRY`, and feature columns)
- `y_train.csv` - Training targets (includes `TARGET` column)
- `X_test.csv` - Test features (same schema as `X_train.csv`)

## Running Models

### XGBoost (single model)
```bash
python supp_material/xgboost_model.py
```

### TabPFN
```bash
python supp_material/TabPFN.py
```

### XGBoost (double model - separate DE/FR)
```bash
python supp_material/double_model_XGB.py
```

### TabPFN (double model)
```bash
python supp_material/double_model_TabPFN.py
```

## Architecture

### Preprocessing Pipeline (`supp_material/preprocessing.py`)

Main function: `data_preprocessing(X_train, Y_train, X_test, convert_categorical=True)`

Feature engineering steps (in order):
1. `find_max_exchange_days()` - Creates `is_market_decupled` boolean using 95th percentile threshold on `DE_FR_EXCHANGE` and `FR_DE_EXCHANGE`
2. `add_total_energy_columns()` - Aggregates energy production by type (COAL, GAS, HYDRO, NUCLEAR, SOLAR, WINDPOW, LIGNITE) to create `DE_TOTAL_ENERGY`, `FR_TOTAL_ENERGY`, and `*_energy_leftover` (production - consumption)
3. `find_holiday_features()` - Creates `HOLIDAY` boolean based on bottom 33% quantile of global consumption days
4. `aggregate_fossil_energy()` - Sums fossil fuel columns into `DE_FOSSIL_ENERGY` and `FR_FOSSIL_ENERGY`
5. `drop_id_column()` - Removes `ID`, `DAY_ID`, `DE_FR_EXCHANGE`, `FR_DE_EXCHANGE`
6. `transform_in_categorical_()` - Converts `COUNTRY` to categorical type if `convert_categorical=True`

Commented-out transformations (available but not active):
- `fill_missing_countries()` - Infers missing COUNTRY values based on DAY_ID
- `find_day_of_the_week()` - Creates day-of-week and month features from DAY_ID
- `aggregate_renewable_energy()` - Sums renewable columns
- `fill_nan_entries()` - Fills NaN with column means
- `transform_into_gaussian()` - Applies Yeo-Johnson power transform
- `drop_by_correlation()` - Removes features with <2.5% Spearman correlation to TARGET

### Double Model Preprocessing (`supp_material/preprocessing_double_model.py`)

Function: `data_preprocessing_double_model(X_train, Y_train, X_test, country, convert_to_categorical=True)`

Key difference from single-model preprocessing:
- `remove_country_specific_columns()` - Drops country-specific features (e.g., drops `FR_*` columns for DE model, except `*_CONSUMPTION`)
- Uses `find_max_exchange_days()` with strict max equality check instead of 95th percentile
- `find_holiday_features()` takes country parameter and uses that country's consumption only (30% quantile threshold)

### XGBoost Model (`supp_material/xgboost_model.py`)

Main function: `launch_xgb(X_train, y_train, X_test, run_test, task_type, cv_folds, params_constrained, holdout_eval, holdout_size, param_grid)`

Key functions:
- `perform_cross_validation()` - GridSearchCV with Spearman correlation scorer for regression tasks
- `evaluate_model()` - Reports training performance and optional 80/20 holdout evaluation
- `create_country_interaction_constraints()` - Creates interaction constraints separating DE/FR features (not actively used in main block)

Model configuration:
- Uses `enable_categorical=True` for categorical COUNTRY column
- Default scoring: Spearman correlation (via `make_scorer(spearman_scorer)`)
- Returns predictions as DataFrame with `predictions` column

### TabPFN Model (`supp_material/TabPFN.py`)

Workflow:
1. Preprocess data with `convert_categorical=False` (TabPFN doesn't handle pandas categorical)
2. 80/20 holdout evaluation on training data
3. Train on full training set using `TabPFNRegressor.create_default_for_version(ModelVersion.V2)`
4. Save predictions via `save_results_to_csv()`

### Double Model Architecture

`double_model_XGB.py` and `double_model_TabPFN.py` train separate models for DE and FR:
1. Split raw data by COUNTRY
2. Preprocess each split separately using country-specific preprocessing
3. Train separate models via cross-validation
4. Combine predictions aligned to original test indices

### Utilities (`supp_material/utils.py`)

`save_results_to_csv(dataset_wit_IDs, predictions, file_name)`:
- Creates timestamped CSV with `ID` and `TARGET` columns
- Handles 1D/2D prediction arrays
- Default output: current directory

## Common Patterns

1. **Hyperparameter tuning**: Modify `param_grid` in `__main__` blocks. XGBoost uses GridSearchCV with Spearman scoring.

2. **Preprocessing changes**: Edit functions in `preprocessing.py` or `preprocessing_double_model.py`. Many transformations are commented out and can be enabled by uncommenting relevant lines in `data_preprocessing()`.

3. **Feature engineering**: All feature creation happens in preprocessing modules. Common operations:
   - Energy aggregations (total, fossil, renewable)
   - Derived features (leftover energy, holidays, market decoupling)
   - Temporal features (day of week, month - currently commented out)

4. **Model evaluation**:
   - Cross-validation score reported during training
   - Optional 80/20 holdout evaluation (set `holdout_eval=True`)
   - Primary metric: Spearman correlation (higher is better)

5. **Output format**: All prediction scripts save to timestamped CSV files with `ID` and `TARGET` columns using `utils.save_results_to_csv()`.
