import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, make_scorer
from scipy.stats import spearmanr
import warnings
from preprocessing import data_preprocessing

warnings.filterwarnings('ignore')


def create_country_interaction_constraints(feature_names):
    """
    Create interaction constraints that separate DE and FR features.

    Features with 'DE' can only interact with other 'DE' features.
    Features with 'FR' can only interact with other 'FR' features.
    Shared features (like GAS_RET, COAL_RET) can interact with everything.

    Args:
        feature_names: List of feature column names

    Returns:
        List of lists, where each inner list contains indices of features
        that can interact with each other
    """
    # Find indices for each group
    de_indices = [i for i, name in enumerate(feature_names) if 'DE' in name.upper()]
    fr_indices = [i for i, name in enumerate(feature_names) if 'FR' in name.upper()]

    # Shared features (commodities, country encoding, etc.)
    shared_indices = [
        i for i, name in enumerate(feature_names)
        if 'DE' not in name.upper() and 'FR' not in name.upper()
    ]

    # Create constraint groups
    # Each group can interact internally
    constraints = []

    # Group 1: DE features (only those not in FR)
    de_only_indices = [i for i in de_indices if i not in fr_indices]
    if de_only_indices:
        constraints.append(feature_names[de_only_indices])

    # Group 2: FR features (only those not in DE)
    fr_only_indices = [i for i in fr_indices if i not in de_indices]
    if fr_only_indices:
        constraints.append(feature_names[fr_only_indices])

    return constraints


def spearman_scorer(y_true, y_pred):
    """
    Calculate Spearman correlation coefficient for use as a scoring metric.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    float
        Spearman correlation coefficient
    """
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation


def define_param_grid(task_type: str = 'regression') -> dict:
    """
    Define the parameter grid for hyperparameter tuning.

    Parameters:
    -----------
    task_type : str
        Type of task: 'regression' or 'classification'

    Returns:
    --------
    dict
        Parameter grid for GridSearchCV
    """
    param_grid = {
        'n_estimators': [5000, ],
        'max_depth': [5, 7, 8],
        'learning_rate': [0.01, 0.001, 0.0005],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.8, 1.0],
        # 'min_child_weight': [1, 3, 5]
    }

    return param_grid


def perform_cross_validation(X_train: pd.DataFrame,
                             y_train: pd.DataFrame,
                             task_type: str = 'regression',
                             cv_folds: int = 5,
                             interaction_costrain=None,
                             param_grid=None) -> tuple:
    """
    Perform cross-validation and hyperparameter tuning.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training targets
    task_type : str
        Type of task: 'regression' or 'classification'
    cv_folds : int
        Number of cross-validation folds

    Returns:
    --------
    tuple
        (best_model, best_params, best_score)
    """
    # Initialize model based on task type
    if task_type == 'regression':
        base_model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            enable_categorical=True,
            # early_stopping_rounds=10,
            interaction_constraints=interaction_costrain
        )
        # Use Spearman correlation as scoring metric
        scoring = make_scorer(spearman_scorer, greater_is_better=True)
    else:
        base_model = XGBClassifier(random_state=42, n_jobs=-1)
        scoring = 'accuracy'

    # Get parameter grid
    if param_grid is None:
        param_grid = define_param_grid(task_type)

    # Perform grid search with cross-validation
    print(f"Starting {cv_folds}-fold cross-validation...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    grid_search.fit(X_train, y_train.values.ravel())

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def print_model_performance(best_params: dict,
                            best_score: float,
                            task_type: str = 'regression') -> None:
    """
    Print the best parameters and cross-validation score.

    Parameters:
    -----------
    best_params : dict
        Best parameters found by cross-validation
    best_score : float
        Best cross-validation score
    task_type : str
        Type of task: 'regression' or 'classification'
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    if task_type == 'regression':
        print(f"\nBest CV Score (Spearman Correlation): {best_score:.4f}")
    else:
        print(f"\nBest CV Score (Accuracy): {best_score:.4f}")

    print("=" * 60 + "\n")


def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   task_type: str = 'regression',
                   holdout_eval: bool = True,
                   holdout_size: float = 0.2,
                   random_state: int = 42) -> None:
    """
    Evaluate model performance. If holdout_eval=True, train a fresh model on 80% of the
    data using the best params and report performance on the remaining 20%.
    """
    # Standard in-sample evaluation on full training data
    y_pred_train = model.predict(X_train)
    print("Training Set Performance:")
    if task_type == 'regression':
        spearman_corr, spearman_pval = spearmanr(y_pred_train, y_train)
        mse = mean_squared_error(y_train, y_pred_train)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_train, y_pred_train)
        print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    else:
        accuracy = accuracy_score(y_train, y_pred_train)
        print(f"  Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_train, y_pred_train))
    print()

    # Optional: fresh training on 80% and evaluation on 20%
    if holdout_eval:
        print("Holdout (80/20) evaluation with fresh training using best parameters...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=holdout_size,
            random_state=random_state,
            shuffle=True
        )

        # Rebuild a fresh model with the same best params
        best_params = model.get_params()
        FreshModelCls = XGBRegressor if task_type == 'regression' else XGBClassifier
        fresh_model = FreshModelCls(**best_params)

        # Fit and evaluate
        fresh_model.fit(X_tr, np.ravel(y_tr))
        y_val_pred = fresh_model.predict(X_val)

        if task_type == 'regression':
            holdout_spear = spearmanr(y_val, y_val_pred).correlation
            print(f"Spearman correlation on the holdout set: {holdout_spear:.4f}")
        else:
            holdout_acc = accuracy_score(y_val, y_val_pred)
            print(f"Accuracy on the holdout set: {100 * holdout_acc:.1f}%")
        print()


def launch_xgb(X_train: pd.DataFrame,
               y_train: pd.DataFrame,
               X_test: pd.DataFrame = None,
               run_test: bool = False,
               task_type: str = 'regression',
               cv_folds: int = 5,
               params_constrained=None,
               holdout_eval: bool = True,
               holdout_size: float = 0.2,
               param_grid=None) -> pd.DataFrame:
    """
    Main function to train XGBoost model with cross-validation.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features dataframe
    y_train : pd.DataFrame
        Training targets dataframe
    X_test : pd.DataFrame, optional
        Test features dataframe
    run_test : bool, default=False
        Whether to generate predictions for test set
    task_type : str, default='regression'
        Type of task: 'regression' or 'classification'
    cv_folds : int, default=5
        Number of cross-validation folds

    Returns:
    --------
    pd.DataFrame or None
        Predictions for test set if run_test=True, else None
    """
    print("Starting XGBoost Model Training Pipeline...")
    print("-" * 60)

    # Perform cross-validation
    best_model, best_params, best_score = perform_cross_validation(
        X_train,
        y_train,
        task_type=task_type,
        cv_folds=cv_folds,
        interaction_costrain=params_constrained,
        param_grid=param_grid
    )

    # Print performance
    print_model_performance(best_params, best_score, task_type)

    # Evaluate on training set (and optional 80/20 holdout)
    evaluate_model(best_model, X_train, y_train, task_type,
                   holdout_eval=holdout_eval, holdout_size=holdout_size)

    # Generate test predictions if requested
    if run_test and X_test is not None:
        print("Generating predictions for test set...")
        test_predictions = best_model.predict(X_test)

        predictions_df = pd.DataFrame({
            'predictions': test_predictions
        })

        print(f"Test predictions generated: {len(predictions_df)} samples")
        return predictions_df

    return None


# Example usage:
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_regression, make_classification

    X_train = pd.read_csv('X_train.csv')
    Y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')



    param_grid = {
        'n_estimators': [500, 5000, 60000],
        'max_depth': [3, 4, 7],
        'learning_rate': [0.05, 0.001, ],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.8, 1.0],
        # 'min_child_weight': [1, 3, 5]
    }

    X_train_processed, X_test_processed = data_preprocessing(X_train=X_train, Y_train=Y_train, X_test=X_test,
                                                             convert_categorical=True)

    predictions = launch_xgb(
        X_train=X_train_processed,
        y_train=Y_train['TARGET'],
        X_test=X_test_processed,
        run_test=True,
        task_type='regression',  # or 'classification'
        cv_folds=10,
        # params_constrained=interaction_constraints,
        param_grid=param_grid
    )
