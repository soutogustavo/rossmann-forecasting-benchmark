"""Data I/O"""

import logging

import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def load_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from S3 and return it as pandas DataFrames.

    Args:
        cfg (PipelineConfig): Configuration for the pipeline.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames with the sales and stores data.
    """
    cutoff = pd.Timestamp(cfg.cutoff_date)

    sales = pd.read_parquet(
        f"{cfg.data_uri}/data-raw/",
        filters=[("Date", "<=", cutoff)]
    )
    stores = pd.read_parquet(f"{cfg.data_uri}/data-store/")

    logger.info("Loaded %d sales rows and %d stores (cutoff=%s)",
                len(sales), len(stores), cfg.cutoff_date
    )
    return sales, stores
