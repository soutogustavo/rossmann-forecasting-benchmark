"""Validation utils for retraining pipeline"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


REQUIRED_SALES_COLUMNS = {
    "Store",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday"
}

REQUIRED_STORE_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance"
}


class DataValidationError(Exception):
    """Data validation error"""


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def validate_data(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    cfg: PipelineConfig
) -> None:
    """Validate data

    Args:
        sales (pd.DataFrame): DataFrame with the sales data.
        stores (pd.DataFrame): DataFrame with the store data.
        cfg (PipelineConfig): Configuration for the pipeline.

    Raises:
        DataValidationError: If the data is not valid.
    """

    errors: list[str] = []

    missing_sales_cols = REQUIRED_SALES_COLUMNS - set(sales.columns)
    missing_store_cols = REQUIRED_STORE_COLUMNS - set(stores.columns)

    if missing_sales_cols or missing_store_cols:
        raise DataValidationError(
            f"Missing columns. sales: {sorted(missing_sales_cols)}, "
            f"stores: {sorted(missing_store_cols)}"
        )

    sales["Date"] = pd.to_datetime(sales["Date"])

    last_day = sales["Date"].max().date()
    if last_day < cfg.cutoff_date:
        errors.append(
            f"Stale data: last date is {last_day}, expected {cfg.cutoff_date}"
        )

    n_dup = int(sales.duplicated(["Store", "Date"]).sum())
    if n_dup:
        errors.append(f"{n_dup} duplicated (store, date) rows")

    for col in ("Store", "Date", "Sales", "Open"):
        n_null = int(sales[col].isna().sum())
        if n_null:
            errors.append(f"{n_null} nulls in '{col}'")

    if errors:
        raise DataValidationError(
            "Data validation failed:\n  - " + "\n  - ".join(errors)
        )

    logger.info("Data validation passed")
