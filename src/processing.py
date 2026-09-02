"""Rossmann processing utils"""

import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans

from src.preprocessing import (
    create_store_profile,
    preprocess_store_profile_data,
)


def calculate_store_factors(model: Prophet, train_data: pd.DataFrame, window_size: int = 60):
    """Calculate store factors for the Rossmann dataset.

    Args:
        model (Prophet): The Prophet model.
        train_data (pd.DataFrame): Training data.
        window_size (int, optional): Window size for the rolling statistics. Defaults to 60.

    Returns:
        dict: Dictionary with store factors.
    """

    stores = train_data["Store"].unique()

    store_factors = {}
    for sid in stores:

        store_data = train_data[train_data["Store"] == sid].copy()
        store_data = store_data.sort_values("Date").tail(window_size).reset_index(drop=True)
        store_data.rename(columns={"Date":"ds", "Sales": "y"}, inplace=True)
        store_data['cap'] = float(store_data['y'].max() * 1.5)
        store_data['floor'] = 0.0

        factor_data = store_data.head(window_size // 2)

        features = ["ds", "cap", "floor"] + list(model.extra_regressors.keys())
        factor_forecast = model.predict(factor_data[features])

        store_factor = (
            factor_data["y"].values /
            factor_forecast["yhat"].values
        ).mean()

        store_factors[int(sid)] = float(abs(store_factor))

    return store_factors


def cluster_rossmann_stores(
    data: pd.DataFrame, store_data: pd.DataFrame,
    num_clusters: int = 9, target_feature: str = "Sales",
    n_init:int=10, random_state:int=42
):
    """Cluster rossmann stores

    Args:
        data (pd.DataFrame): Training data.
        store_data (pd.DataFrame): Store data.
        num_clusters (int, optional): Number of clusters. Defaults to 9.
        target_feature (str, optional): Target feature. Defaults to "Sales".
        n_init (int, optional): Number of initializations for KMeans. Defaults to 10.
        random_state (int, optional): Random state for KMeans. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame with the added cluster feature.
    """
    store_features = create_store_profile(data=data, store_data=store_data)

    X_processed = preprocess_store_profile_data(profile_data=store_features)

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=n_init
    )
    store_features["Cluster"] = kmeans.fit_predict(X_processed)

    return store_features
