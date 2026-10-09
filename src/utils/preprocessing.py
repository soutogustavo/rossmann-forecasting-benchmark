"""Module for preprocessing the Rossmann dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.feature_engineering import (
    create_lag_features,
    create_rolling_features,
    same_weekday_lags,
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

    cluster_x_train = cluster_ptrain[cluster_features]

    agg_x = cluster_x_train.groupby("ds", observed=True).agg({
        "y": "mean",  # Sales
        "Open": "mean",
        "Promo": "mean",
        "DayOfWeek": "max"
    }).reset_index().round(3)
    agg_x["floor"] = 0

    return (agg_x, cluster_ptrain)


def split_cluster_data(
    train_size: float,
    agg_x: pd.DataFrame,
    margin: float = 1.15
):
    """Splits the cluster data into training and testing sets.

    Args:
        train_size (float): The size of the training set.
        agg_x (pd.DataFrame): The aggregated cluster data.
        margin (float): The margin to apply to the maximum sales by day.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.Series):
            The training and testing sets and cap by day.
    """

    num_points = int(agg_x.shape[0] * train_size)
    agg_x_train = agg_x[:num_points]
    agg_x_test = agg_x[num_points:]

    cap_by_day = agg_x_train.groupby("DayOfWeek", observed=True)["y"].max() * margin
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


def preprocess_store_profile_data(
    profile_data: pd.DataFrame,
    target_feature:str="sales"
):
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

    df["is_holiday"] = np.where(
        (df["stateholiday"] != "0") |
        (df["schoolholiday"] == 1),
    1, 0)

    df = create_lag_features(
        data=df,
        target=target_feature,
        lags=same_weekday_lags(horizon_size),
        horizon=horizon_size,
        id_feature="store",
        date_feature="date"
    )
    df = create_rolling_features(
        data=df,
        target=target_feature,
        horizon=horizon_size,
        windows=(7, 28),
        id_feature="store",
        date_feature="date"
    )

    return df


def split_input_data(
    data: pd.DataFrame,
    cutoff,
    horizon: int = 60
):
    """Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): Dataset.
        cutoff (pd.Timestamp): Cutoff date.
        horizon (int): Horizon size.

    Returns:
        tuple: Train, validation, and test sets.
    """
    T = pd.Timestamp(cutoff)

    holdout_start = T - pd.Timedelta(days=horizon - 1)    # T-59
    es_start = holdout_start - pd.Timedelta(days=horizon)  # T-119

    d = data["date"]

    train   = data[d < es_start]
    es      = data[(d >= es_start) & (d < holdout_start)]
    holdout = data[(d >= holdout_start) & (d <= T)]

    return train, es, holdout


def set_cluster_feature(
    data: pd.DataFrame,
    store_profile: pd.DataFrame,
    store_id: str = "store"
):
    """Set cluster feature to the dataset.

    Args:
        data (pd.DataFrame): Dataset.
        store_profile (pd.DataFrame): Store profile data.
        store_id (str, optional): Store id column name. Defaults to "store".

    Returns:
        pd.DataFrame: Dataset with cluster feature.
    """
    df = data.copy()
    df_store = store_profile.copy()

    store_cluster_map = df_store[[store_id, "cluster"]].drop_duplicates()
    store_cluster_map.set_index(store_id, inplace=True)
    store_cluster_map = store_cluster_map.to_dict()["cluster"]

    df["cluster"] = df[store_id].map(lambda x: store_cluster_map[x])
    df["cluster"] = df["cluster"].astype("category")

    return df


def build_schema(X_train: pd.DataFrame) -> dict:
    """Build schema from training data.

    Args:
        X_train (pd.DataFrame): Training data.

    Returns:
        dict: Dictionary with schema.
    """
    return {
        "columns": list(X_train.columns),
        "dtypes": X_train.dtypes.to_dict()
    }
