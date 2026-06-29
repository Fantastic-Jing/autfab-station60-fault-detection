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

from config import DATA_DIR, TEST_FILE_MARKERS, TRAIN_STRIDE_STEPS, RANDOM_SEED, WINDOW_STEPS
from preprocessing import process_file, resample_file, extract_windows_first_fault

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
    def _collect(file_list: list[str], mode: str = "train") -> tuple[np.ndarray, np.ndarray]:
        """Collect processed windows from file_list.

        mode: "train" or "test". Passed to preprocessing to control windowing.
        """
        X_parts, y_parts = [], []
        for fp in file_list:
            try:
                if mode == "train":
                    # First pass: original Pure Window behaviour
                    X_pure, y_pure = process_file(fp, mode="train")

                    # Second pass: First Fault Label windows using TRAIN_STRIDE_STEPS
                    try:
                        features_rs, labels_rs, _ = resample_file(fp)
                        X_ffl, y_ffl = extract_windows_first_fault(
                            features_rs, labels_rs, WINDOW_STEPS, TRAIN_STRIDE_STEPS
                        )
                    except Exception as e:
                        logger.error("Failed first-fault extraction for %s: %s", fp, e)
                        X_ffl, y_ffl = np.empty((0,)), np.empty((0,))

                    # Downsample ffl windows to match number of pure windows (1:1) if needed
                    n_pure = len(X_pure) if hasattr(X_pure, "shape") else 0
                    if n_pure > 0 and len(X_ffl) > n_pure:
                        rng = np.random.default_rng(RANDOM_SEED)
                        # Downsample First Fault Label windows to a 4:1 ratio vs pure windows.
                        # Keep at least one sample when n_pure is small.
                        idx = rng.choice(len(X_ffl), size=max(1, n_pure // 4), replace=False)
                        X_ffl = X_ffl[idx]
                        y_ffl = y_ffl[idx]

                    # Combine both passes
                    if (len(X_pure) == 0) and (len(X_ffl) == 0):
                        logger.warning("No windows extracted from %s in both passes; skipping.", fp)
                        continue
                    elif len(X_pure) == 0:
                        X_combined, y_combined = X_ffl, y_ffl
                    elif len(X_ffl) == 0:
                        X_combined, y_combined = X_pure, y_pure
                    else:
                        X_combined = np.concatenate([X_pure, X_ffl], axis=0)
                        y_combined = np.concatenate([y_pure, y_ffl], axis=0)

                    X_parts.append(X_combined)
                    y_parts.append(y_combined)
                else:
                    # Test mode: use existing process_file behaviour (First Fault Label + TEST_STRIDE_STEPS)
                    X, y = process_file(fp, mode=mode)
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
    X_train, y_train = _collect(train_files, mode="train")
    logger.info("Processing test files ...")
    X_test, y_test = _collect(test_files, mode="test")

    # Fit on training data only;
    # but apply the same statistics to the test set.
    scaler = ChannelStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Optionally show transformed samples
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
