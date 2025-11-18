import  pandas as pd
from scipy.stats import spearmanr
import numpy as np
from sklearn.preprocessing import PowerTransformer


def metric_train(output, Y_clean):
    return 100 * (spearmanr(output, Y_clean).correlation)


def drop_by_correlation(X_train, Y_train, X_test):
    correlation = [metric_train(X_train[col], Y_train['TARGET']) for col in X_train.columns]
    correlation_df = pd.DataFrame(correlation, index=X_train.columns, columns=['Features'])

    correlation_df.sort_values(by='Features').plot(kind='bar', figsize=(10, 5))

    low_corr = correlation_df[abs(correlation_df['Features']) < 2.5]

    columns_to_keep = ['COUNTRY', 'COUNTRY_DE', 'COUNTRY_FR', "is_market_decupled"]
    columns_to_drop = [column for column in low_corr.index if column not in columns_to_keep]
    X_dropped_train = X_train.drop(columns=columns_to_drop, errors='ignore')

    X_dropped_test = X_test.drop(columns=columns_to_drop, errors='ignore')

    return X_dropped_train, X_dropped_test




def transform_in_categorical_(X_train, X_test):
    categorical_cols = ["COUNTRY",  ] #"month_number" ,"day_of_week",

    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X_train, X_test

def find_max_exchange_days(X):
    df = X.copy()
    # Use 95th percentile (highest 5%) instead of max
    max_de_fr = df["DE_FR_EXCHANGE"].dropna().quantile(0.95)
    max_fr_de = df["FR_DE_EXCHANGE"].dropna().quantile(0.95)



    df["is_market_decupled"] = ~(
            df["DE_FR_EXCHANGE"].ge(max_de_fr) |
            df["FR_DE_EXCHANGE"].ge(max_fr_de)
    )

    return df

def find_day_of_the_week(X):
    df = X.copy()
    # Compute day of week (0–6)
    df["day_of_week"] = df["DAY_ID"] % 7
    # Compute month index (approximate: every 30 days)
    #df["month_number"] = (df["DAY_ID"] // 30)


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

    # If you want to see which DAY_IDs were filled:
    # print(filled_rows.sort_values('DAY_ID'))


    return df
def aggregate_renewable_energy(df: pd.DataFrame) -> pd.DataFrame:
    renewable_suffixes = ["HYDRO", "SOLAR", "WINDPOW"]
    out = df.copy()
    out["DE_RENEWABLE_ENERGY"] = out[[f"DE_{s}" for s in renewable_suffixes]].sum(axis=1, min_count=1)
    out["FR_RENEWABLE_ENERGY"] = out[[f"FR_{s}" for s in renewable_suffixes]].sum(axis=1, min_count=1)
    #out["TOTAL_RENEWABLE_ENERGY"] = out["DE_RENEWABLE_ENERGY"] + out["FR_RENEWABLE_ENERGY"]
    #out = out.drop(columns=[f"DE_{s}" for s in renewable_suffixes] + [f"FR_{s}" for s in renewable_suffixes])
    return out

def aggregate_fossil_energy(df: pd.DataFrame) -> pd.DataFrame:

    fossil_suffixes_de= ["GAS", "COAL",  "NUCLEAR" ,"LIGNITE"]
    fossil_suffixes_fr= ["GAS", "COAL",  "NUCLEAR",]
    out = df.copy()
    out["DE_FOSSIL_ENERGY"] = out[[f"DE_{s}" for s in fossil_suffixes_de]].sum(axis=1, min_count=1)
    out["FR_FOSSIL_ENERGY"] = out[[f"FR_{s}" for s in fossil_suffixes_fr]].sum(axis=1, min_count=1)
    #out = out.drop(columns=[f"DE_{s}" for s in renewable_suffixes_de] + [f"FR_{s}" for s in renewable_suffixes_fr])
    return out

def drop_id_column(X_train, X_test):
    columns_to_drop = ['ID', 'DAY_ID', 'DE_FR_EXCHANGE', 'FR_DE_EXCHANGE']
    X_train_dropped = X_train.copy()
    X_test_dropped = X_test.copy()
    for col in columns_to_drop:
        X_train_dropped = X_train_dropped.drop(columns=[col])
        X_test_dropped = X_test_dropped.drop(columns=[col])
    return X_train_dropped, X_test_dropped



def add_total_energy_columns(df: pd.DataFrame) -> pd.DataFrame:

    prod_suffixes_de = ["GAS", "COAL", "HYDRO", "NUCLEAR", "SOLAR", "WINDPOW", "LIGNITE"]
    prod_suffixes_fr = ["GAS", "COAL", "HYDRO", "NUCLEAR", "SOLAR", "WINDPOW"]
    out = df.copy()
    out["DE_TOTAL_ENERGY"] = out[[f"DE_{s}" for s in prod_suffixes_de]].sum(axis=1, min_count=1)
    out["FR_TOTAL_ENERGY"] = out[[f"FR_{s}" for s in prod_suffixes_fr]].sum(axis=1, min_count=1)


    out["DE_energy_leftover"] = out["DE_TOTAL_ENERGY"] - out["DE_CONSUMPTION"]
    out["FR_energy_leftover"] = out["FR_TOTAL_ENERGY"] - out["FR_CONSUMPTION"]

    return out



def find_holiday_features(X):
    # df is your dataframe with DAY_ID, DE_CONSUMPTION / FR_CONSUMPTION (or ...CONSUMPTON)


    df = X.copy()
    n_unique_days = df["DAY_ID"].nunique()


    # --- 2) Global consumption (robust to CONSUMPTON typo)
    de_cons = df.get("DE_CONSUMPTION")
    fr_cons = df.get("FR_CONSUMPTION")


    df["GLOBAL_CONSUMPTION"] = pd.to_numeric(de_cons, errors="coerce") + pd.to_numeric(fr_cons, errors="coerce")

    # --- 3) Sum by day
    by_day = df.groupby("DAY_ID", as_index=False)["GLOBAL_CONSUMPTION"].sum()

    # --- 4) Bottom 0.33 quantile threshold
    q33 = by_day["GLOBAL_CONSUMPTION"].quantile(0.33)
    low_days = set(by_day.loc[by_day["GLOBAL_CONSUMPTION"] <= q33, "DAY_ID"])

    # --- 5) Mark HOLIDAY on original df (True if day is in bottom-quantile set)
    df["HOLIDAY"] = df["DAY_ID"].isin(low_days)


    return df

def transform_into_gaussian(X):
    df= X.copy()

    # Columns to exclude
    exclude_cols = ['ID', 'DAY_ID', 'COUNTRY', 'COUNTRY_DE', 'COUNTRY_FR', 'is_market_decupled', 'HOLIDAY']
    exclude_cols = [col for col in exclude_cols if col in df.columns]

    # Select numeric columns except excluded ones
    cols_to_transform = df.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore').columns

    # Apply PowerTransformer only to selected columns
    pt = PowerTransformer(method='yeo-johnson')
    transformed = pt.fit_transform(df[cols_to_transform])

    # Create DataFrame for transformed columns
    df_transformed = pd.DataFrame(transformed, columns=cols_to_transform, index=df.index)

    # Combine transformed and untouched columns
    df_final = pd.concat([df_transformed, df[exclude_cols]], axis=1)


    return df_final

def fill_nan_entries(X):
    df = X.copy()
    df = df.drop(columns = ['COUNTRY'])
    df = df.fillna(df.mean())
    df['COUNTRY'] = X['COUNTRY']
    return df


def data_preprocessing(X_train, Y_train, X_test, convert_categorical: bool = True):
    #X_train = fill_missing_countries(X_train)
    #X_test = fill_missing_countries(X_test)
    X_train = find_max_exchange_days(X_train)
    X_test = find_max_exchange_days(X_test)
    #X_train = find_day_of_the_week(X_train)
    #X_test = find_day_of_the_week(X_test)
    X_test = add_total_energy_columns(X_test)
    X_train = add_total_energy_columns(X_train)
    X_test = find_holiday_features(X_test)
    X_train = find_holiday_features(X_train)
    X_train = aggregate_fossil_energy(X_train)
    X_test = aggregate_fossil_energy(X_test)
    X_train = aggregate_renewable_energy(X_train)
    X_test = aggregate_renewable_energy(X_test)
    #X_train = fill_nan_entries(X_train)
    #X_test = fill_nan_entries(X_test)




    X_train, X_test = drop_id_column(X_train, X_test)

    #X_train = transform_into_gaussian(X_train)
    #X_test = transform_into_gaussian(X_test)

    if convert_categorical:
        X_train, X_test = transform_in_categorical_(X_train, X_test)
    #X_train, X_test = drop_by_correlation(X_train, Y_train, X_test)


    return X_train, X_test