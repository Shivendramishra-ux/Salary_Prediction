"""
utils.py
--------
Shared utilities used across the Salary Prediction project:
    - Centralized logging configuration
    - Project-wide constants (paths, column names)
    - Small reusable helper functions

Keeping these in one place avoids duplicated logic in
preprocess.py, train.py, evaluate.py, and predict.py.
"""

import logging
import os
import sys

# ----------------------------------------------------------------------
# Project paths (relative to project root)
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "salary.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "salary_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "project.log")

# ----------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------
TARGET_COLUMN = "Salary"
NUMERICAL_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Gender", "Education Level", "Job Title", "City"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger that writes to both
    the console and a shared log file (logs/project.log).

    Parameters
    ----------
    name : str
        Usually __name__ of the calling module.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is fetched multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs() -> None:
    """Create all directories the project needs, if they don't already exist."""
    for directory in [MODEL_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)
