import pandas as pd
from tabpfn.constants import ModelVersion
from tabpfn import TabPFNRegressor

try:
    from .preprocessing_double_model import data_preprocessing_double_model
except Exception:
    from preprocessing_double_model import data_preprocessing_double_model
# NEW: save helper
try:
    from .utils import save_results_to_csv
except Exception:
    from utils import save_results_to_csv


def launch_double_model(X_train: pd.DataFrame,
                        Y_train: pd.DataFrame,
                        X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Train two TabPFN models:
    - One on rows where COUNTRY == 'DE'
    - One on rows where COUNTRY == 'FR'
    Always return stitched predictions for X_test.
    """
    test_predictions = pd.Series(index=X_test.index, dtype=float)

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
            X_train_country_raw, y_train_country_raw, X_test_country_raw, country=country_code, convert_to_categorical=False
        )

        # Drop COUNTRY and align y (in case preprocessing filtered rows)
        X_train_country_feat = X_train_country_proc.drop(columns=["COUNTRY"], errors="ignore")
        y_train_country = y_train_country_raw.reindex(X_train_country_proc.index)['TARGET']

        # Train TabPFN (v2) and predict
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        model.fit(X_train_country_feat, y_train_country)

        if not X_test_country_proc.empty:
            X_test_country_feat = X_test_country_proc.drop(columns=["COUNTRY"], errors="ignore")
            test_predictions.loc[X_test_country_proc.index] = model.predict(X_test_country_feat)

    return pd.DataFrame({"predictions": test_predictions})


if __name__ == "__main__":
    X_train = pd.read_csv('X_train.csv')
    Y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')

    print("Launching per-country TabPFN training and prediction...")
    test_preds_df = launch_double_model(X_train=X_train, Y_train=Y_train, X_test=X_test)

    # Save the results using the shared utility (keeps ID alignment)
    save_results_to_csv(
        dataset_wit_IDs=X_test,
        predictions=test_preds_df['predictions'].values,
        file_name='double_model_test_predictions'
    )
