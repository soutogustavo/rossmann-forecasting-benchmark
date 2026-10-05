"""Module for feature engineering for the Rossmann dataset."""

import numpy as np
import pandas as pd


def create_lags_features_rossmann(
    data: pd.DataFrame, target_feature: str,
    num_lags:int=3, id_feature:str="store", min_shift_lag:int=7
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


def create_rolling_stats_features_rossmann(
    data: pd.DataFrame,
    target_feature:str,
    window_size:int=3,
    id_feature:str="store",
    min_shift_lag:int=7
):
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
    df[f"safe_{target_feature}"] = df.groupby(id_feature)[
        target_feature].shift(min_shift_lag)

    df[f"{target_feature}_mean"] = df.groupby(id_feature)[
        f"safe_{target_feature}"
    ].rolling(window_size).mean().values

    df[f"{target_feature}_sum"] = df.groupby(id_feature)[
        f"safe_{target_feature}"
    ].rolling(window_size).sum().values

    df[f"{target_feature}_std"] = df.groupby(id_feature)[
        f"safe_{target_feature}"
    ].rolling(window_size).std().values

    df.drop(columns=[f"safe_{target_feature}"], inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def create_time_based_features_rossmann(
    data: pd.DataFrame,
    date_feature:str="date"
):
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


def create_ewm_features_rossmann(
    data: pd.DataFrame,
    target_feature:str,
    id_feature:str="store",
    span:int=7
):
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

    df["ewm_mean"] = df.groupby(id_feature)[
        target_feature].ewm(span=span).mean().values

    return df


def create_seasonal_indicator_features_rossmann(
    data: pd.DataFrame,
    date_feature:str="date"
):
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


def create_cyclical_time_features_rossmann(
    data: pd.DataFrame,
    cyclical_features:list
):
    """Create cyclical features with sine and cosine.

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        cyclical_features (list): List of features to use for cyclical encoding.

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.copy()

    dow_feature = "dayofweek"
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
