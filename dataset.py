"""
Dataset assembly:
  - Scans DATA_DIR for CSV files and assigns each file to train or test split
    based on TEST_FILE_MARKERS in config.py.
  - Processes all files through the preprocessing pipeline.
  - Fits a per-channel standard scaler (z-score normalization) on the training set only,
    then applies the same transform to the test set.
"""

import os
import logging
import numpy as np

from config import DATA_DIR, TEST_FILE_MARKERS
from preprocessing import process_file

logger = logging.getLogger(__name__)



# File-level split


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



# Normalization (z-score per channel)


class ChannelStandardScaler:
    """
    Per-channel standard scaler (z-score normalization) for 3-D arrays.
    fit() is called on training data; transform() is applied to both splits.
    Handles constant channels and very small standard deviations.
    """

    def __init__(self):
        self.ch_mean: np.ndarray | None = None  # shape (n_channels,)
        self.ch_std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ChannelStandardScaler":
        # Compute mean/std over samples and time axis, keeping channel axis
        self.ch_mean = X.mean(axis=(0, 2))      # (n_channels,)
        self.ch_std = X.std(axis=(0, 2))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.ch_mean is not None, "Scaler has not been fitted yet."
        std = self.ch_std.copy()
        # Handle both constant channels and very small standard deviations
        std[std < 1e-7] = 1.0
        # Broadcast: (n_channels,) -> (1, n_channels, 1)
        mean_ = self.ch_mean[np.newaxis, :, np.newaxis]
        std_ = std[np.newaxis, :, np.newaxis]
        return (X - mean_) / std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)



# Dataset builder


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
    scaler = ChannelStandardScaler()
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
