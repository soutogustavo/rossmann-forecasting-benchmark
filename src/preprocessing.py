"""Module for preprocessing the Rossmann dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    df["Store"] = df["Store"].astype("category")

    # Check the difference between competitor and Rossmann store
    # Negative values mean Rossmann store opened a store before its the competitor
    df["days_to_competitor"] = (
        df["Date"] -
        pd.to_datetime(df["CompetitionOpenSinceYear"], format="%Y")).dt.days

    df.sort_values(["Store", "Date"], inplace=True)

    return df


def preprocess_rossmann_store_data(data: pd.DataFrame):
    """Applies preprocessing to the Rossmann store dataset.

    Args:
        data (pd.DataFrame): The dataset to preprocess.

    Returns:
        pd.DataFrame: The dataset with added features.
    """
    data["Store"] = data["Store"].astype("category")

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


def create_lags_features_rossmann(
    data: pd.DataFrame, target_feature: str,
    num_lags:int=3, id_feature:str="Store", min_shift_lag:int=7
):
    """Generate lags features for the Rossmann dataset give a target feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        target_feature (str): Name of the target feature.
        num_lags (int): Number of lags to generate.
        id_feature (str): Name of the id feature.
        min_shift_lag (int): Minimum shift lag to generate.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    # The minimum shift for any autoregressive feature was locked at $t-7$,
    # aligning with a real-world weekly corporate planning cycle.
    for lag in range(min_shift_lag, num_lags+min_shift_lag+1):
        df[f"{target_feature}_{lag}"] = df.groupby(id_feature)[target_feature].shift(lag)

    df.reset_index(drop=True, inplace=True)

    return df


def create_rolling_stats_features_rossmann(data: pd.DataFrame, target_feature:str, window_size:int=3, id_feature:str="Store", min_shift_lag:int=7):
    """Create rolling stats (mean, sum, std) given a target feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        target_feature (str): Name of the target feature.
        window_size (int): Window size for the rolling statistics.
        id_feature (str): Name of the id feature.
        min_shift_lag (int): Minimum shift lag to generate.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    # Rolling statistics (Mean, Sum, Std) were computed exclusively over
    # the shifted $t-7$ baseline, eliminating any look-ahead bias.
    df[f"safe_{target_feature}"] = df.groupby(id_feature)[target_feature].shift(min_shift_lag)

    df[f"{target_feature}_mean"] = df.groupby(id_feature)[f"safe_{target_feature}"].rolling(window_size).mean().values
    df[f"{target_feature}_sum"] = df.groupby(id_feature)[f"safe_{target_feature}"].rolling(window_size).sum().values
    df[f"{target_feature}_std"] = df.groupby(id_feature)[f"safe_{target_feature}"].rolling(window_size).std().values

    df.drop(columns=[f"safe_{target_feature}"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def create_time_based_features_rossmann(data: pd.DataFrame, date_feature:str="Date"):
    """Create time-based features given date feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        date_feature (str): Name of the date feature.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    df["day_of_week"] = df[date_feature].dt.day_name().values
    df["month"] = df[date_feature].dt.month.values
    df["month_name"] = df[date_feature].dt.month_name().values
    df["week_year"] = df[date_feature].dt.isocalendar().week.values
    df["quarter"] = df[date_feature].dt.quarter.values

    df.reset_index(drop=True, inplace=True)

    return df


def create_ewm_features_rossmann(data: pd.DataFrame, target_feature:str, id_feature:str="Store", span:int=7):
    """Create a exponentially weighted moving average features given target feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        target_feature (str): Name of the target feature.
        id_feature (str): Name of the id feature.
        span (int): Span for the exponentially weighted moving average.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    df["ewm_mean"] = df.groupby(id_feature)[target_feature].ewm(span=span).mean().values

    return df


def create_seasonal_indicator_features_rossmann(data: pd.DataFrame, date_feature:str="Date"):
    """Create seasonal indicators given date feature

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        date_feature (str): Name of the date feature.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    df["is_weekend"] = df[date_feature].dt.dayofweek >= 5
    df["is_month_start"] = df[date_feature].dt.day <= 5
    df["is_month_end"] = df[date_feature].dt.day >= 26

    return df


def create_cyclical_time_features_rossmann(data: pd.DataFrame, cyclical_features:list):
    """Create cyclical features with sine and cosine.

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        cyclical_features (list): List of features to use for cyclical encoding.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    dow_feature = "DayOfWeek"
    if dow_feature in cyclical_features:
        df[f"{dow_feature}_sin"] = np.sin(2 * np.pi * df[dow_feature] / 7)
        df[f"{dow_feature}_cos"] = np.cos(2 * np.pi * df[dow_feature] / 7)

    month_feature = "month"
    if "month" in cyclical_features:
        df[f"{month_feature}_sin"] = np.sin(2 * np.pi * df[month_feature] / 12)
        df[f"{month_feature}_cos"] = np.cos(2 * np.pi * df[month_feature] / 12)

    woy_feature = "week_year"
    if "month" in cyclical_features:
        df[f"{woy_feature}_sin"] = np.sin(2 * np.pi * df[woy_feature] / 52)
        df[f"{woy_feature}_cos"] = np.cos(2 * np.pi * df[woy_feature] / 52)

    return df


def create_store_profile(
    data: pd.DataFrame,
    store_data: pd.DataFrame,
    target_feature:str="Sales",
    id_feature:str="Store"
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

    store_features = df.groupby(id_feature).agg({
        target_feature: "mean",
        "Customers": "mean",
        "CompetitionDistance": "mean"
    }).reset_index()

    store_features = store_features.merge(
        df_store[[id_feature, "StoreType"]],
        on=id_feature
    )

    store_features.columns = [
        id_feature,
        f"Avg{target_feature}",
        "AvgCustomers",
        "AvgCompetitionDistance",
        "StoreType"
    ]

    return store_features


def preprocess_store_profile_data(profile_data: pd.DataFrame, target_feature:str="Sales"):
    """Preprocess the store profile data.

    Args:
        profile_data (pd.DataFrame): DataFrame with the store profile.

    Returns:
        pd.DataFrame: DataFrame with the processed store profile.
    """
    df = profile_data.copy()

    num_features = [f"Avg{target_feature}", "AvgCustomers"]
    cat_features = ["StoreType"]

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
    target_feature: str = "Sales",
    id_feature: str = "Store",
    categorical_features=None
):

    df = data.copy()
    df_profile = profile_data.copy()

    #categorical_cols = ['StoreType', 'Assortment', 'Cluster']

    clustered_train = df.merge(
        df_profile[[id_feature, "Cluster"]],
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
    train_encoded["IsHoliday"] = np.where(
        (train_encoded["StateHoliday"] != "0") |
        (train_encoded["SchoolHoliday"] == 1),
    1, 0)

    drop_features = [
        "StateHoliday", "SchoolHoliday",
        "Customers", "PromoInterval",
        "Promo2SinceWeek", "Promo2SinceYear"
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
    date_feature:str="Date",
    target_feature:str="Sales",
    id_feature: str = "Store"
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
        [date_feature, target_feature, "Customers"]
    ]

    X_train = e_train[features]
    y_train = e_train[target_feature]

    X_val = e_val[features]
    y_val = e_val[target_feature]

    return X_train, y_train, X_val, y_val
