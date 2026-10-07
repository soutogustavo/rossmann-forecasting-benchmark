"""Module for preprocessing the Rossmann dataset."""

import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.feature_engineering import (
    create_lags_features_rossmann,
    create_rolling_stats_features_rossmann,
)


def merge_sales_and_store_data(
    sales: pd.DataFrame, store: pd.DataFrame, id_feature:str="Store"):
    """Merges the sales and store datasets.

    Args:
        sales (pd.DataFrame): DataFrame with the sales data.
        store (pd.DataFrame): DataFrame with the store data.
        id_feature (str): Name of the id feature to merge the datasets.

    Returns:
        pd.DataFrame: DataFrame with the merged data.
    """

    return sales.merge(store, on=id_feature)


def preprocess_rossmann_data(df, max_competitor_distance: int = 200000):
    """Applies preprocessing to the Rossmann dataset.

    Args:
        df (pd.DataFrame): The dataset to preprocess.
        max_competitor_distance (int): The maximum distance to a competitor.

    Returns:
        pd.DataFrame: The dataset with added features.
    """

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["competitiondistance"] = df["competitiondistance"].fillna(
        max_competitor_distance)

    df["competitionopenmonth"] = df.apply(
        lambda x: x["date"].month
        if np.isnan(x["competitionopensincemonth"])
        else x["competitionopensincemonth"], axis=1
    )

    df["competitionopenyear"] = df.apply(
        lambda x: x["date"].year
        if np.isnan(x["competitionopensinceyear"])
        else x["competitionopensinceyear"], axis=1
    )

    binary_features = ["promo", "promo2", "schoolholiday"]
    df[binary_features] = df[binary_features].fillna(0)

    df["open"] = np.where(df["open"].isna(), 0, df["open"])

    df["store"] = df["store"].astype("category")

    # Check the difference between competitor and Rossmann store
    # Negative values mean Rossmann store opened a store before its the competitor
    df["days_to_competitor"] = (
        df["date"] -
        pd.to_datetime(df["competitionopensinceyear"], format="%Y")).dt.days

    df.sort_values(["store", "date"], inplace=True)

    return df


def preprocess_rossmann_store_data(data: pd.DataFrame):
    """Applies preprocessing to the Rossmann store dataset.

    Args:
        data (pd.DataFrame): The dataset to preprocess.

    Returns:
        pd.DataFrame: The dataset with added features.
    """

    data.columns = data.columns.str.lower()
    data["store"] = data["store"].astype("category")

    return data


def prepare_cluster_data_for_training(
    store_features: pd.DataFrame,
    ptrain: pd.DataFrame,
    cluster_id: int,
    cluster_features: list
):
    """Prepares cluster data for training.

    Args:
        store_features (pd.DataFrame): DataFrame with store features.
        ptrain (pd.DataFrame): DataFrame with training data.
        cluster_id (int): The cluster ID.
        cluster_features (list): List of features to use for training.

    Returns:
        pd.DataFrame: The aggregated cluster data.
    """

    cluster_stores = store_features[store_features["Cluster"] == cluster_id]
    cluster_ptrain = ptrain[ptrain["Store"].isin(cluster_stores["Store"])]
    cluster_ptrain.sort_values(["Store", "Date"], inplace=True)
    cluster_ptrain.reset_index(drop=True, inplace=True)
    cluster_ptrain.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)

    #cluster_features = ["ds", "y", "Open", "Promo", "DayOfWeek"]
    cluster_x_train = cluster_ptrain[cluster_features]

    agg_x = cluster_x_train.groupby("ds").agg({
        "y": "mean",  # Sales
        "Open": "mean",
        "Promo": "mean",
        "DayOfWeek": "max"
    }).reset_index().round(3)
    agg_x["floor"] = 0

    return (agg_x, cluster_ptrain)


def split_cluster_data(train_size: float, agg_x: pd.DataFrame, margin: float = 1.15):
    """Splits the cluster data into training and testing sets.

    Args:
        train_size (float): The size of the training set.
        agg_x (pd.DataFrame): The aggregated cluster data.
        margin (float): The margin to apply to the maximum sales by day.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.Series): The training and testing sets and cap by day.
    """

    num_points = int(agg_x.shape[0] * train_size)
    agg_x_train = agg_x[:num_points]
    agg_x_test = agg_x[num_points:]

    cap_by_day = agg_x_train.groupby("DayOfWeek")["y"].max() * margin
    cap_by_day = cap_by_day.replace(0.0, 0.0001)

    agg_x_train['cap'] = agg_x_train["DayOfWeek"].map(cap_by_day)
    agg_x_test['cap'] = agg_x_test["DayOfWeek"].map(cap_by_day)

    return (agg_x_train, agg_x_test, cap_by_day)


def create_store_profile(
    data: pd.DataFrame,
    store_data: pd.DataFrame,
    target_feature:str="sales",
    id_feature:str="store"
):
    """Create a store profile given a target feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        store_data (pd.DataFrame): DataFrame with the store data.
        target_feature (str): Name of the target feature.
        id_feature (str): Name of the id feature.

    Returns:
        pd.DataFrame: DataFrame with the store profile.
    """

    df = data.copy()
    df_store = store_data.copy()

    store_features = df.groupby(id_feature, observed=True).agg({
        target_feature: "mean",
        "customers": "mean",
        "competitiondistance": "mean"
    }).reset_index()

    store_features = store_features.merge(
        df_store[[id_feature, "storetype"]],
        on=id_feature
    )

    store_features.columns = [
        id_feature,
        f"avg_{target_feature}",
        "avg_customers",
        "avg_competition_distance",
        "storetype"
    ]

    return store_features


def preprocess_store_profile_data(profile_data: pd.DataFrame, target_feature:str="sales"):
    """Preprocess the store profile data.

    Args:
        profile_data (pd.DataFrame): DataFrame with the store profile.

    Returns:
        pd.DataFrame: DataFrame with the processed store profile.
    """
    df = profile_data.copy()

    num_features = [f"avg_{target_feature}", "avg_customers"]
    cat_features = ["storetype"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features)
        ]
    )

    return preprocessor.fit_transform(df)


def extended_preprocessing_rossmann_xgb(
    data: pd.DataFrame,
    profile_data: pd.DataFrame,
    target_feature: str = "sales",
    id_feature: str = "store",
    categorical_features=None
):

    df = data.copy()
    df_profile = profile_data.copy()

    #categorical_cols = ['storetype', 'assortment', 'cluster']

    clustered_train = df.merge(
        df_profile[[id_feature, "cluster"]],
        on=id_feature,
        how="left"
    )

    if categorical_features is not None:
        train_encoded = pd.get_dummies(
            clustered_train,
            columns=categorical_features,
            drop_first=True
        )
    else:
        train_encoded = clustered_train

    # Transform StateHoliday and SchoolHoliday into one column called IsHoliday
    train_encoded["is_holiday"] = np.where(
        (train_encoded["stateholiday"] != "0") |
        (train_encoded["schoolholiday"] == 1),
    1, 0)

    drop_features = [
        "stateholiday", "schoolholiday",
        "customers", "promointerval",
        "promo2sinceweek", "promo2sinceyear"
    ]
    train_encoded.drop(columns=drop_features, inplace=True)

    train_encoded = create_lags_features_rossmann(
        data=train_encoded,
        target_feature=target_feature
    )
    train_encoded = create_rolling_stats_features_rossmann(
        data=train_encoded,
        target_feature=target_feature
    )

    return train_encoded


def split_train_val_data(
    data: pd.DataFrame,
    date_threshold:str,
    date_feature:str="date",
    target_feature:str="sales",
    id_feature: str = "store"
):
    """Split train and validation data based on date threshold

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        date_threshold (str): Date threshold for splitting the data.
        target_feature (str): Name of the target feature.
        id_feature (str): Name of the id feature.

    Returns:
        tuple: Tuple containing the train and validation data.
    """

    df = data.copy()

    e_train = df[df[date_feature] < date_threshold]
    e_val = df[df[date_feature] >= date_threshold]

    features = [col for col in e_train.columns if col not in
        [date_feature, target_feature, "customers"]
    ]

    X_train = e_train[features]
    y_train = e_train[target_feature]

    X_val = e_val[features]
    y_val = e_val[target_feature]

    # Copy for naive evaluation
    X_val_naive = X_val.copy()
    X_val_naive["date"] = e_val[date_feature]
    X_val_naive[target_feature] = y_val.values

    return X_train, y_train, X_val, y_val, X_val_naive


def split_initial_data(
    data: pd.DataFrame,
    date_threshold:str,
    date_feature:str="date",
    horizon_size: int = 60
):
    """Split initial data (train + validation) from data based on date threshold

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        date_threshold (str): Date threshold for splitting the data.

    Returns:
        tuple: Tuple containing the initial data (train + validation).
    """

    df = data.copy()

    train = df[df[date_feature] < date_threshold]
    val = df[df[date_feature] >= date_threshold]

    horizon_end = pd.Timestamp(date_threshold) + timedelta(days=horizon_size-1)

    return train, val[val[date_feature] <= horizon_end]


def preprocessing_for_xgb(
    data: pd.DataFrame,
    horizon_size: int = 60,
    target_feature: str = "sales",
    categorical_features = None
):
    """Preprocess the data for XGBoost model.

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        horizon_size (int): Horizon size for the model.
        target_feature (str): Name of the target feature.
        categorical_features (list): List of categorical features.

    Returns:
        pd.DataFrame: DataFrame with the preprocessed data.
    """

    df = data.copy()

    if categorical_features is not None:
        df[categorical_features] = df[categorical_features].astype("category")

    # Transform StateHoliday and SchoolHoliday into one column called IsHoliday
    df["is_holiday"] = np.where(
        (df["stateholiday"] != "0") |
        (df["schoolholiday"] == 1),
    1, 0)

    '''drop_features = [
        "stateholiday", "schoolholiday",
        "customers", "promointerval",
        "promo2sinceweek", "promo2sinceyear"
    ]
    df.drop(columns=drop_features, inplace=True)'''

    df = create_lags_features_rossmann(
        data=df,
        min_shift_lag=horizon_size,
        target_feature=target_feature
    )
    df = create_rolling_stats_features_rossmann(
        data=df,
        min_shift_lag=horizon_size,
        target_feature=target_feature
    )

    return df

# ----------

def three_way_split(df: pd.DataFrame, cutoff, horizon: int = 60):
    T = pd.Timestamp(cutoff)

    holdout_start = T - pd.Timedelta(days=horizon - 1)    # T-59
    es_start = holdout_start - pd.Timedelta(days=horizon)  # T-119

    d = df["date"]

    train   = df[d < es_start]
    es      = df[(d >= es_start) & (d < holdout_start)]
    holdout = df[(d >= holdout_start) & (d <= T)]

    return train, es, holdout


def set_clusters(train: pd.DataFrame, store_profile: pd.DataFrame, store_id: str = "store"):
    df = train.copy()
    df_store = store_profile.copy()

    store_cluster_map = df_store[["store", "cluster"]].drop_duplicates()
    store_cluster_map.set_index("store", inplace=True)
    store_cluster_map = store_cluster_map.to_dict()["cluster"]

    df["cluster"] = df["store"].map(lambda x: store_cluster_map[x])
    df["cluster"] = df["cluster"].astype("category")

    return df


def build_schema(X_train: pd.DataFrame) -> dict:
    return {
        "columns": list(X_train.columns),
        "dtypes": X_train.dtypes.to_dict()
    }
