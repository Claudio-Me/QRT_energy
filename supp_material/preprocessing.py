
import  pandas as pd
from scipy.stats import spearmanr
import numpy as np

def metric_train(output, Y_clean):
    return 100 * (spearmanr(output, Y_clean).correlation)


def drop_by_correlation(X_train, Y_train, X_test):
    correlation = [metric_train(X_train[col], Y_train['TARGET']) for col in X_train.columns]
    correlation_df = pd.DataFrame(correlation, index=X_train.columns, columns=['Features'])

    correlation_df.sort_values(by='Features').plot(kind='bar', figsize=(10, 5))

    low_corr = correlation_df[abs(correlation_df['Features']) < 2.5]

    columns_to_keep = ['COUNTRY', 'COUNTRY_DE', 'COUNTRY_FR', "is_market_decupled"]
    columns_to_drop = [column for column in low_corr.index if column not in columns_to_keep]
    X_dropped_train = X_train.drop(columns=columns_to_drop)

    X_dropped_test = X_test.drop(columns=columns_to_drop)

    return X_dropped_train, X_dropped_test




def transform_in_categorical_(X_train, X_test):
    categorical_cols = ["COUNTRY", "day_of_week", ] #"month_number"

    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X_train, X_test

def find_max_exchange_days(X):
    df = X.copy()
    # assuming df is your DataFrame
    max_de_fr = df["DE_FR_EXCHANGE"].max(skipna=True)
    max_fr_de = df["FR_DE_EXCHANGE"].max(skipna=True)

    df["is_market_decupled"] = ~(
            df["DE_FR_EXCHANGE"].eq(max_de_fr) |
            df["FR_DE_EXCHANGE"].eq(max_fr_de)
    )

    return df

def find_day_of_the_week(X):
    df = X.copy()
    # Compute day of week (0–6)
    df["day_of_week"] = df["DAY_ID"] % 7
    # Compute month index (approximate: every 30 days)
    df["month_number"] = (df["DAY_ID"] // 30)


    return df

def fill_missing_countries(X):
    df= X.copy()
    # df has columns: ID, DAY_ID, COUNTRY (values 'FR'/'DE' or NaN)
    mask_na = df["COUNTRY"].isna()

    # For each DAY_ID, get the single known country if (and only if) there's exactly one unique known value
    known = df.groupby("DAY_ID")["COUNTRY"].transform(
        lambda s: s.dropna().unique()[0] if s.dropna().nunique() == 1 else np.nan
    )

    # Map to the opposite country
    opp = known.map({"FR": "DE", "DE": "FR"})

    # Fill only where COUNTRY is NaN and we have a determinate opposite
    to_fill = mask_na & opp.notna()
    df.loc[to_fill, "COUNTRY"] = opp[to_fill].values

    # (Optional) report what was filled and what remains missing
    filled_rows = df.loc[to_fill, ["ID", "DAY_ID", "COUNTRY"]]
    remaining_missing = df["COUNTRY"].isna().sum()
    print(f"Filled {to_fill.sum()} rows. Remaining missing COUNTRY: {remaining_missing}")
    # If you want to see which DAY_IDs were filled:
    # print(filled_rows.sort_values('DAY_ID'))


    return df

def drop_id_column(X_train, X_test):
    columns_to_drop = ['ID']
    for col in columns_to_drop:
        X_train_dropped = X_train.drop(columns=[col])
        X_test_dropped = X_test.drop(columns=[col])
    return X_train_dropped, X_test_dropped

def data_preprocessing(X_train, Y_train, X_test):
    X_train = fill_missing_countries(X_train)
    X_test = fill_missing_countries(X_test)
    X_train = find_max_exchange_days(X_train)
    X_test = find_max_exchange_days(X_test)
    X_train = find_day_of_the_week(X_train)
    X_test = find_day_of_the_week(X_test)
    X_train, X_test = drop_id_column(X_train, X_test)

    #X_train, X_test = transform_in_categorical_(X_train, X_test)
    #X_train, X_test = drop_by_correlation(X_train, Y_train, X_test)


    return X_train, X_test