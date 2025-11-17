import pandas as pd
from xgboost import XGBRegressor

try:
    from .preprocessing_double_model import data_preprocessing_double_model
except Exception:
    from preprocessing_double_model import data_preprocessing_double_model

try:
    from .xgboost_model import perform_cross_validation, print_model_performance
except Exception:
    from xgboost_model import perform_cross_validation, print_model_performance
# NEW: save helper
try:
    from .utils import save_results_to_csv
except Exception:
    from utils import save_results_to_csv


def launch_double_model(X_train: pd.DataFrame,
                        Y_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        cv_folds: int = 5,
                        param_grid=None) -> pd.DataFrame:
    """
    Train two XGBoost models with CV (DE and FR) and always return test predictions.
    Preprocessing is applied after splitting by COUNTRY.
    """
    # Final predictions aligned to X_test
    test_predictions = pd.Series(index=X_test.index, dtype=float)

    country_models = {}
    train_columns_by_country = {}

    for country_code in ["DE", "FR"]:
        # Split raw data by COUNTRY
        train_mask = X_train["COUNTRY"].astype(str) == country_code
        test_mask = X_test["COUNTRY"].astype(str) == country_code
        if not train_mask.any():
            print(f"Skip {country_code}: no training rows.")
            continue

        X_train_country_raw = X_train.loc[train_mask].copy()
        y_train_country_raw = Y_train.loc[train_mask].copy()
        X_test_country_raw = X_test.loc[test_mask].copy()

        # Preprocess only the country-specific splits
        X_train_country_proc, X_test_country_proc = data_preprocessing_double_model(
            X_train_country_raw, y_train_country_raw, X_test_country_raw, country=country_code
        )

        # Drop COUNTRY for modeling
        X_train_country_feat = X_train_country_proc.drop(columns=["COUNTRY"], errors="ignore")
        # Align y to processed X indices in case preprocessing filtered rows
        y_train_country = y_train_country_raw.reindex(X_train_country_proc.index)

        print(f"\n=== Cross-validation for COUNTRY == {country_code} ===")
        best_model, best_params, best_score = perform_cross_validation(
            X_train_country_feat, y_train_country['TARGET'], task_type='regression', cv_folds=cv_folds,
            param_grid=param_grid
        )
        print_model_performance(best_params, best_score, task_type='regression')

        country_models[country_code] = best_model
        train_columns_by_country[country_code] = list(X_train_country_feat.columns)

        # Predict on the processed test split (if any)
        if not X_test_country_proc.empty:
            X_test_country_feat = (
                X_test_country_proc
                .drop(columns=["COUNTRY"], errors="ignore")
                .reindex(columns=train_columns_by_country[country_code], fill_value=0)
            )
            test_predictions.loc[X_test_country_proc.index] = best_model.predict(X_test_country_feat)

    return pd.DataFrame({"predictions": test_predictions})


if __name__ == "__main__":
    X_train = pd.read_csv('X_train.csv')
    Y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')

    print("Launching double model training and prediction...")
    param_grid = {
        'n_estimators': [400, 5000, ],
        'max_depth': [5, 7, 8],
        'learning_rate': [0.01, 0.001, 0.0005],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.8, 1.0],
        # 'min_child_weight': [1, 3, 5]
    }

    test_preds_df = launch_double_model(X_train=X_train, Y_train=Y_train, X_test=X_test, cv_folds=5,
                                        param_grid=param_grid)

    # Save the results using the shared utility (keeps ID alignment)
    save_results_to_csv(
        dataset_wit_IDs=X_test,
        predictions=test_preds_df['predictions'].values,
        file_name='double_model_test_predictions'
    )
