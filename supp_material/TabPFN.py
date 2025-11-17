from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tabpfn.constants import ModelVersion
from scipy.stats import spearmanr
import pandas as pd

from tabpfn import TabPFNRegressor
from supp_material.utils import save_results_to_csv
from supp_material.preprocessing import data_preprocessing

if __name__ == "__main__":
    X_train = pd.read_csv('X_train.csv')
    Y_train = pd.read_csv('Y_train.csv')
    X_test = pd.read_csv('X_test.csv')

    print("Preprocessing training data...")
    X_train_processed, X_test_processed = data_preprocessing(X_train=X_train, Y_train=Y_train, X_test=X_test)
    print(len(X_train_processed))
    # X_train_processed.head()



    # Split indices to keep alignment
    X_80_train, X_20_test, y_80_train, y_20_test = train_test_split(
        X_train_processed, Y_train, test_size=0.2, random_state=41, shuffle=True
    )

    # test performances
    regressor_train = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.

    regressor_train = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

    # To use TabPFN v2:
    # regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    regressor_train.fit(X_80_train, y_80_train['TARGET'])

    # Predict on the test set
    predictions_train = regressor_train.predict(X_20_test)

    holdout_spear = spearmanr(y_20_test['TARGET'], predictions_train).correlation
    print(f"Spearman correlation on the holdout set: {holdout_spear:.4f}")

    regressor = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.
    # To use TabPFN v2:
    # regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
    regressor.fit(X_train_processed, Y_train['TARGET'])

    # Predict on the test set
    predictions = regressor.predict(X_test_processed)

    # Save the results in a csv file
    save_results_to_csv(dataset_wit_IDs=X_test, predictions=predictions, file_name='tabpfn_test_performances')

