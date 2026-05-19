"""Module for preprocessing the Rossmann dataset."""

import numpy as np
import pandas as pd


def preprocess_rossmann_data(df, max_competitor_distance: int = 200000):
    """Applies preprocessing to the Rossmann dataset.

    Args:
        df (pd.DataFrame): The dataset to preprocess.
        max_competitor_distance (int): The maximum distance to a competitor.

    Returns:
        pd.DataFrame: The dataset with added features.
    """

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        max_competitor_distance)

    df["CompetitionOpenSinceMonth"] = df.apply(
        lambda x: x["Date"].month
        if np.isnan(x["CompetitionOpenSinceMonth"])
        else x["CompetitionOpenSinceMonth"], axis=1
    )

    df["CompetitionOpenSinceYear"] = df.apply(
        lambda x: x["Date"].year
        if np.isnan(x["CompetitionOpenSinceYear"])
        else x["CompetitionOpenSinceYear"], axis=1
    )

    binary_features = ["Promo", "Promo2", "SchoolHoliday"]
    df[binary_features] = df[binary_features].fillna(0)

    df["Open"] = np.where(df["Open"].isna(), 0, df["Open"])

    # Check the difference between competitor and Rossmann store
    # Negative values mean Rossmann store opened a store before its the competitor
    df["days_to_competitor"] = (
        df["Date"] -
        pd.to_datetime(df["CompetitionOpenSinceYear"], format="%Y")).dt.days

    return df


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
