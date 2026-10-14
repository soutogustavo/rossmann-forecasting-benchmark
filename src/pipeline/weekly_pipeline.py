"""Weekly job: retrain and forecast."""

import logging
import sys

from src.pipeline import inference_pipeline, retrain_pipeline

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    retrain_code = retrain_pipeline.main(argv)

    if retrain_code == 1:
        logger.error("Retrain failed: forecast will not be executed")
        return 1
    if retrain_code == 2:
        logger.warning("Candidate rejected in gate: forecast with the current champion")

    return inference_pipeline.main(argv)


if __name__ == "__main__":
    sys.exit(main())
