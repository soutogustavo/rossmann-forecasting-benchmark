"""Rossmann processing utils"""

import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.cluster import KMeans

from src.utils.evaluation import RossmannEvaluation, smape
from src.utils.preprocessing import (
    create_store_profile,
    preprocess_store_profile_data,
)


def calculate_store_factors(
    model: Prophet,
    train_data: pd.DataFrame,
    window_size: int = 60
):
    """Calculate store factors for the Rossmann dataset.

    Args:
        model (Prophet): The Prophet model.
        train_data (pd.DataFrame): Training data.
        window_size (int, optional): Window size for the rolling statistics. Defaults to 60.

    Returns:
        dict: Dictionary with store factors.
    """

    stores = train_data["store"].unique()

    store_factors = {}
    for sid in stores:

        store_data = train_data[train_data["store"] == sid].copy()
        store_data = store_data.sort_values("date").tail(window_size).reset_index(drop=True)
        store_data.rename(columns={"date":"ds", "sales": "y"}, inplace=True)
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
    num_clusters: int = 9, target_feature: str = "sales",
    n_init:int=10, random_state:int=42
):
    """Cluster rossmann stores

    Args:
        data (pd.DataFrame): Training data.
        store_data (pd.DataFrame): Store data.
        num_clusters (int, optional): Number of clusters. Defaults to 9.
        target_feature (str, optional): Target feature. Defaults to "sales".
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
    store_features["cluster"] = kmeans.fit_predict(X_processed)

    return store_features


def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    rossmann_eval: RossmannEvaluation
):
    """Train the XGBoost model"""

    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
        tree_method="hist",
        enable_categorical=True,
        eval_names=["train", "val"],
        eval_metric=rossmann_eval.smape_adjusted,
    )

    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        verbose=50
    )

    # Post-processing
    preds = model_xgb.predict(X_val)
    preds = np.where(X_val['open'] == 0, 0, preds)

    pred_smape = smape(y_true=y_val, y_pred=preds)
    print(f"sMAPE = {pred_smape:.3f}")


    return model_xgb


def build_model(model_params: dict):
    return xgb.XGBRegressor(**model_params)
