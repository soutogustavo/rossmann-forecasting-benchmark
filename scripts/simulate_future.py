"""Replay (only dev): saves future covariates from a cutoff to data-future/.

Does the role of the company's planning system: from the lines of the
train after T, keeps only what would be known in advance.

Usage:
    python -m scripts.simulate_future --cutoff-date 2015-05-31
"""

import argparse
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config

# No Sales/Customers: only what is known before the day happens
FUTURE_COLUMNS = [
    "Store", "Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday",
]


def future_from_train(
    train: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int
) -> pd.DataFrame:
    """Lines from T+1 to T+H, only with the allowed columns.

    Args:
        train (pd.DataFrame): Training data.
        cutoff (pd.Timestamp): Cutoff date.
        horizon (int): Forecast horizon.

    Returns:
        pd.DataFrame: Future covariates.
    """
    window = (
        (train["Date"] > cutoff) &
        (train["Date"] <= cutoff + pd.Timedelta(days=horizon))
    )
    return train.loc[window, FUTURE_COLUMNS].copy()


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Simulates future covariates (replay)"
    )
    parser.add_argument("--env", default="dev")
    parser.add_argument("--cutoff-date", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.env, args.cutoff_date)
    if cfg.env != "dev":
        raise SystemExit("Simulate future not implemented for production")

    T = pd.Timestamp(cfg.cutoff_date)
    train = pd.read_parquet(f"{cfg.data_uri}/data-raw/")
    train["Date"] = pd.to_datetime(train["Date"])
    future = future_from_train(train, T, cfg.horizon_days)

    n_stores = future["Store"].nunique()
    n_days = future["Date"].nunique()
    if n_days != cfg.horizon_days or len(future) != n_stores * n_days:
        raise SystemExit(
            f"Incomplete grid: {len(future)} rows, {n_stores} stores x {n_days} days "
            f"(expected {cfg.horizon_days} days)"
        )

    path = f"{cfg.data_uri}/data-future/cutoff={T:%Y-%m-%d}/covariates.parquet"
    future.to_parquet(path, index=False)
    print(f"{len(future)} rows ({n_stores} stores x {n_days} days) -> {path}")


if __name__ == "__main__":
    main()
