"""Module for feature engineering for the Rossmann dataset."""

import numpy as np
import pandas as pd


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

    df["ewm_mean"] = df.groupby(id_feature, observed=True)[
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


def same_weekday_lags(horizon: int, n: int = 4) -> list[int]:
    """Generate lags of the same weekday.
        Every lag needs to be >= horizon (no look-ahead).

    Args:
        horizon (int): Horizon for the lags.
        n (int, optional): Number of lags to generate. Defaults to 4.

    Returns:
        list[int]: List of lags.
    """
    first = -(-horizon // 7) * 7
    return [first + 7 * k for k in range(n)]


def create_lag_features(
    data: pd.DataFrame,
    target: str,
    lags: list[int],
    *,
    horizon: int,
    id_feature: str = "store",
    date_feature: str = "date",
) -> pd.DataFrame:
    """Lags of the target per store. Every lag needs to be >= horizon (no look-ahead).

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        target (str): Name of the target feature.
        lags (list[int]): List of lags to generate.
        horizon (int): Horizon for the lags.
        id_feature (str, optional): Name of the id feature. Defaults to "store".
        date_feature (str, optional): Name of the date feature. Defaults to "date".

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """

    df = data.sort_values([id_feature, date_feature]).copy()
    g = df.groupby(id_feature, observed=True)[target]
    for lag in lags:
        df[f"{target}_lag_{lag}"] = g.shift(lag)
    return df


def create_rolling_features(
    data: pd.DataFrame,
    target: str,
    *,
    horizon: int,
    windows: tuple[int, ...] = (7, 28),
    id_feature: str = "store",
    date_feature: str = "date",
) -> pd.DataFrame:
    """Mean and standard deviation in windows over the series shifted by `horizon` days.

    Args:
        data (pd.DataFrame): DataFrame with the dataset.
        target (str): Name of the target feature.
        windows (tuple[int, ...], optional): Window sizes for the rolling statistics. Defaults to (7, 28).
        id_feature (str, optional): Name of the id feature. Defaults to "store".
        date_feature (str, optional): Name of the date feature. Defaults to "date".

    Returns:
        pd.DataFrame: DataFrame with the added features.
    """
    df = data.sort_values([id_feature, date_feature]).copy()
    if not df.index.is_unique:
        raise ValueError("Index duplication: index alignment would fail")

    shifted = df.groupby(id_feature, observed=True)[target].shift(horizon)
    g = shifted.groupby(df[id_feature], observed=True)
    for w in windows:
        df[f"{target}_roll{w}_mean"] = g.rolling(w).mean().droplevel(0)
        df[f"{target}_roll{w}_std"] = g.rolling(w).std().droplevel(0)
    return df
