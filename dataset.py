"""
Dataset assembly:
  - Scans DATA_DIR for CSV files and assigns each file to train or test split
    based on TEST_FILE_MARKERS in config.py.
  - Processes all files through the preprocessing pipeline.
  - Fits a per-channel min-max scaler on the training set only, then applies
    the same transform to the test set.
"""

import os
import logging
import numpy as np

from config import DATA_DIR, TEST_FILE_MARKERS
from preprocessing import process_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-level split
# ---------------------------------------------------------------------------

def _is_test_file(filename: str) -> bool:
    """Return True if the filename matches any marker in TEST_FILE_MARKERS."""
    for markers in TEST_FILE_MARKERS.values():
        for marker in markers:
            if marker in filename:
                return True
    return False


def split_files(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Scan data_dir and return (train_files, test_files).
    Only .CSV / .csv files are considered.
    """
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ])

    train_files, test_files = [], []
    for fp in all_files:
        fname = os.path.basename(fp)
        if _is_test_file(fname):
            test_files.append(fp)
        else:
            train_files.append(fp)

    logger.info("Split: %d train files, %d test files", len(train_files), len(test_files))
    return train_files, test_files


# ---------------------------------------------------------------------------
# Normalization (min-max, per channel)
# ---------------------------------------------------------------------------

class ChannelMinMaxScaler:
    """
    Per-channel min-max scaler for 3-D arrays (n_samples, n_channels, n_timesteps).
    fit() is called on training data; transform() is applied to both splits.
    """

    def __init__(self):
        self.ch_min: np.ndarray | None = None   # shape (n_channels,)
        self.ch_max: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ChannelMinMaxScaler":
        # Compute min/max over samples and time axis, keeping channel axis
        self.ch_min = X.min(axis=(0, 2))        # (n_channels,)
        self.ch_max = X.max(axis=(0, 2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.ch_min is not None, "Scaler has not been fitted yet."
        rng = self.ch_max - self.ch_min
        rng[rng == 0] = 1.0                     # avoid division by zero for constant channels
        # Broadcast: (n_channels,) -> (1, n_channels, 1)
        min_ = self.ch_min[np.newaxis, :, np.newaxis]
        rng_ = rng[np.newaxis, :, np.newaxis]
        return (X - min_) / rng_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str = DATA_DIR) -> dict:
    """
    Process all CSV files, apply normalisation, and return a dict with:
        X_train, y_train, X_test, y_test  (numpy arrays)
        scaler                             (fitted ChannelMinMaxScaler)
        train_files, test_files            (file path lists, for traceability)
    """
    train_files, test_files = split_files(data_dir)

    def _collect(file_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X_parts, y_parts = [], []
        for fp in file_list:
            try:
                X, y = process_file(fp)
                if len(X) > 0:
                    X_parts.append(X)
                    y_parts.append(y)
                else:
                    logger.warning("No windows extracted from %s; skipping.", fp)
            except Exception as exc:
                logger.error("Failed to process %s: %s", fp, exc)
        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

    logger.info("Processing training files ...")
    X_train, y_train = _collect(train_files)

    logger.info("Processing test files ...")
    X_test, y_test = _collect(test_files)

    # Fit scaler on training data only
    scaler = ChannelMinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    logger.info(
        "Dataset ready — train: %s  test: %s  classes: %s",
        X_train.shape, X_test.shape, np.unique(y_train).tolist()
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
        "scaler":  scaler,
        "train_files": train_files,
        "test_files":  test_files,
    }
