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


def split_files(data_dir: str, show_examples: int = 3) -> tuple[list[str], list[str]]:
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

    # Show a few example file paths from each split for quick verification
    if show_examples and (train_files or test_files):
        n = max(0, int(show_examples))
        logger.info("Example train files: %s", train_files[:n])
        logger.info("Example test  files: %s", test_files[:n])

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
        # (n_channels,) -> (1, n_channels, 1)
        mean_ = self.ch_mean[np.newaxis, :, np.newaxis]
        std_ = std[np.newaxis, :, np.newaxis]
        return (X - mean_) / std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform_sample(self, X: np.ndarray, n_samples: int = 2) -> np.ndarray:
        """Transform X and return the first n_samples of the transformed data.

        Useful to inspect how data looks after scaling without processing the
        whole dataset in the caller.
        """
        assert self.ch_mean is not None, "Scaler has not been fitted yet."
        Xt = self.transform(X)
        return Xt[:max(0, int(n_samples))]

    def describe_transformed(self, X: np.ndarray, n_samples: int = 2, decimals: int = 4) -> None:
        """Log simple statistics (per-channel mean/std) for first n_samples.

        This shows how the scaler changed data at a glance.
        """
        Xt = self.transform(X)
        n = min(int(n_samples), Xt.shape[0]) if Xt.size else 0
        for i in range(n):
            sample = Xt[i]
            means = sample.mean(axis=1)
            stds = sample.std(axis=1)
            logger.info("Transformed sample %d: per-channel mean=%s std=%s",
                        i,
                        np.round(means, decimals).tolist(),
                        np.round(stds, decimals).tolist())



# Dataset builder
def build_dataset(data_dir: str = DATA_DIR, show_samples: int = 3) -> dict:
    """
    Process CSV files, apply normalization, and return a dict with:
        X_train, y_train, X_test, y_test  (numpy arrays)
        scaler                             (fitted ChannelStandardScaler)
        train_files, test_files            (file path lists, for traceability)

        # X: channel values, shape (n_windows, 44, 128) — samples x channels x timesteps
        # y: label for each window, shape (n_windows,)
        # scaler: stores per-channel mean/std fitted on training data; reused to scale test data
        # axis=0: samples dimension;
        # axis=1: channels;
        # axis=2: timesteps

    If show_samples > 0, print a few examples after scaling to verify the output.
    """

    # File-level split to prevent data leakage between adjacent windows.
    train_files, test_files = split_files(data_dir)

    # function: collecting processed each file and stack results into (total_windows, 44, 128).
    def _collect(file_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X_parts, y_parts = [], []
        for fp in file_list:
            try:
                # Call preprocessing function
                X, y = process_file(fp)
                if len(X) > 0:
                    X_parts.append(X)
                    y_parts.append(y)
                else:
                    logger.warning("No windows extracted from %s; skipping.", fp)
            except Exception as exc:
                # Log and continue so one bad file doesn't stop everything.
                logger.error("Failed to process %s: %s", fp, exc)

        return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

    # Process files and collect all windows into final train/test arrays.
    logger.info("Processing training files ...")
    X_train, y_train = _collect(train_files)
    logger.info("Processing test files ...")
    X_test, y_test = _collect(test_files)

    # Fit on training data only; apply the same statistics to the test set.
    scaler = ChannelStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Optionally show a few transformed samples to verify the scaling effect.
    if show_samples and X_train.size:
        n = min(int(show_samples), X_train.shape[0])
        logger.info("Showing %d transformed training samples (per-channel mean/std):", n)
        scaler.describe_transformed(X_train, n_samples=n)
    if show_samples and X_test.size:
        n_test = min(int(show_samples), X_test.shape[0])
        logger.info("Showing %d transformed test samples (per-channel mean/std):", n_test)
        scaler.describe_transformed(X_test, n_samples=n_test)

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
