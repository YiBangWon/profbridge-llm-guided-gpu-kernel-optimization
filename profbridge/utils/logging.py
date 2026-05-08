from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("profbridge")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(stream)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger
