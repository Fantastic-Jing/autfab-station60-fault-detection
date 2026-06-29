"""
Preprocessing pipeline:
  1. Load a CSV file and resolve feature / label columns.
  2. Remap labels (merge transition states into label 1).
  3. Resample each feature channel to TARGET_HZ using polyphase filtering.
  4. Extract fixed-length windows with a Pure Window label strategy.
  5. Zero-pad files that are shorter than one window.
"""

import os
import logging
import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import resample_poly

from config import (
    TIMESTAMP_COLS, LABEL_COL, LABEL_REMAP, VALID_LABELS,
    TARGET_HZ, WINDOW_STEPS, STRIDE_STEPS,
)
from config import TEST_STRIDE_STEPS

logger = logging.getLogger(__name__)

# Internal helpers
def _estimate_source_hz(df: pd.DataFrame) -> float:
    """Estimate sampling frequency from the Second + Nanosecond columns."""
    seconds = df["Second"].astype(float).values
    nanos   = df["Nanosecond"].astype(float).values
    timestamps_ns = seconds * 1_000_000_000 + nanos

    diffs = np.diff(timestamps_ns)
    diffs = diffs[diffs > 0]          # drop zero-diff rows (duplicate timestamps)
    if len(diffs) == 0:
        logger.warning("Could not estimate frequency; defaulting to 64 Hz.")
        return float(TARGET_HZ)

    median_interval_s = np.median(diffs) / 1_000_000_000
    return 1.0 / median_interval_s

def _resample_array(arr: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """
    Resample from src_hz to dst_hz
    Uses the exact rational approximation (up/down integers) for quality.
    """
    if abs(src_hz - dst_hz) < 0.5:   # already close enough
        return arr

    ratio = Fraction(dst_hz / src_hz).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(arr, up, down).astype(np.float32)

def _extract_windows(
    features: np.ndarray,   # shape (n_timesteps, n_channels)
    labels: np.ndarray,     # shape (n_timesteps,)
    window_len: int,
    stride: int,
    mode: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the time axis.

    Modes:
      - "train": Pure Window strategy (keep windows where every timestep has same label).
      - "test" : Keep every window. Assign label = first non-zero label inside window,
                 or 0 if none. This produces denser test windows and does not discard
                 mixed-label windows.

    Returns:
        X: (n_windows, n_channels, window_len)
        y: (n_windows,)
    """
    n_timesteps, n_channels = features.shape
    windows_X, windows_y = [], []

    for start in range(0, n_timesteps - window_len + 1, stride):
        end = start + window_len
        window_labels = labels[start:end]

        if mode == "train":
            # Original Pure Window behaviour: skip mixed-label windows
            unique = np.unique(window_labels)
            if len(unique) != 1:
                continue
            label = int(unique[0])
            if label not in VALID_LABELS:
                continue

        else:  # mode == "test"
            # Keep every window. Assign first non-zero label found, else 0.
            non_zero = window_labels[window_labels != 0]
            label = int(non_zero[0]) if len(non_zero) > 0 else 0
            if label not in VALID_LABELS:
                # If label after remap is out of valid range, skip to avoid invalid targets
                continue

        windows_X.append(features[start:end].T)   # (n_channels, window_len)
        windows_y.append(label)

    if len(windows_X) == 0:
        return np.empty((0, n_channels, window_len), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(windows_X).astype(np.float32), np.array(windows_y, dtype=np.int32)


def extract_windows_first_fault(
    features: np.ndarray,   # shape (n_timesteps, n_channels)
    labels: np.ndarray,     # shape (n_timesteps,)
    window_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide windows with given stride. For each window keep it only if there is a
    non-zero label inside. Assign the first non-zero label as the window label.

    This is intended for the First Fault Label pass used on training files.
    Returns X (n_windows, n_channels, window_len) and y (n_windows,)
    """
    n_timesteps, n_channels = features.shape
    windows_X, windows_y = [], []

    for start in range(0, n_timesteps - window_len + 1, stride):
        end = start + window_len
        window_labels = labels[start:end]
        non_zero = window_labels[window_labels != 0]
        if len(non_zero) == 0:
            # skip windows that contain only label 0
            continue
        label = int(non_zero[0])
        if label not in VALID_LABELS:
            continue
        windows_X.append(features[start:end].T)
        windows_y.append(label)

    if len(windows_X) == 0:
        return np.empty((0, n_channels, window_len), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(windows_X).astype(np.float32), np.array(windows_y, dtype=np.int32)



# Public API
def load_csv(filepath: str) -> pd.DataFrame:
    """Load a Station 60 CSV file; handle separator variants gracefully."""
    try:
        df = pd.read_csv(filepath, sep=";", decimal=",")
    except Exception:
        df = pd.read_csv(filepath, sep=",")

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names: all columns except timestamps and label."""
    exclude = set(TIMESTAMP_COLS) | {LABEL_COL}
    return [c for c in df.columns if c not in exclude]


def resample_file(filepath: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load CSV and resample features and labels to TARGET_HZ.

    Returns (features_rs, labels_rs, src_hz)
    features_rs: shape (n_timesteps_resampled, n_channels)
    labels_rs:   shape (n_timesteps_resampled,)
    src_hz:      estimated source Hz (float)
    """
    df = load_csv(filepath)
    filename = os.path.basename(filepath)

    feat_cols = get_feature_cols(df)
    n_channels = len(feat_cols)

    # Label remapping
    raw_labels = df[LABEL_COL].values.astype(int)
    labels = np.array([LABEL_REMAP.get(l, l) for l in raw_labels], dtype=np.int32)

    # Feature matrix (timesteps x channels), cast to float
    features_raw = df[feat_cols].values.astype(np.float32)

    # Resample each channel independently
    src_hz = _estimate_source_hz(df)

    # Resample first channel to get actual output length
    ch0_resampled = _resample_array(features_raw[:, 0], src_hz, TARGET_HZ)
    actual_len = len(ch0_resampled)

    # Resample remaining channels with length normalization
    features_rs = np.zeros((actual_len, n_channels), dtype=np.float32)
    features_rs[:, 0] = ch0_resampled
    for ch in range(1, n_channels):
        ch_resampled = _resample_array(features_raw[:, ch], src_hz, TARGET_HZ)
        # Handle minor length discrepancies (truncate or pad)
        if len(ch_resampled) > actual_len:
            features_rs[:, ch] = ch_resampled[:actual_len]
        elif len(ch_resampled) < actual_len:
            features_rs[:len(ch_resampled), ch] = ch_resampled
        else:
            features_rs[:, ch] = ch_resampled

    # Resample labels to match actual feature length
    label_indices = np.round(
        np.linspace(0, len(labels) - 1, actual_len)
    ).astype(int)
    labels_rs = labels[label_indices]

    # Zero-pad if shorter than one window
    if actual_len < WINDOW_STEPS:
        pad_len = WINDOW_STEPS - actual_len
        logger.warning(
            "%s: only %d steps after resampling; zero-padding %d steps.",
            filename, actual_len, pad_len
        )
        # Pad with repeated last value for each channel (more realistic than zeros)
        last_values = features_rs[-1:, :]  # (1, n_channels)
        pad_features = np.tile(last_values, (pad_len, 1))
        features_rs = np.vstack([features_rs, pad_features])

        # Pad labels with the dominant label in the file
        valid_labels = labels_rs[labels_rs >= 0]
        if len(valid_labels) > 0:
            dominant_label = int(np.bincount(valid_labels).argmax())
        else:
            dominant_label = 0  # Fallback to label 0 if no valid labels
        labels_rs = np.pad(labels_rs, (0, pad_len), constant_values=dominant_label)

    return features_rs, labels_rs, src_hz

def process_file(filepath: str, mode: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing for a single CSV file.

    Parameters:
        filepath: path to CSV
        mode: "train" or "test". Controls windowing behaviour and stride.

    Returns:
        X: (n_windows, n_channels, WINDOW_STEPS)  float32
        y: (n_windows,)                           int32
    """
    df = load_csv(filepath)
    filename = os.path.basename(filepath)

    # Resolve columns
    feat_cols = get_feature_cols(df)
    n_channels = len(feat_cols)

    # Label remapping
    raw_labels = df[LABEL_COL].values.astype(int)
    labels = np.array([LABEL_REMAP.get(l, l) for l in raw_labels], dtype=np.int32)

    # Feature matrix (timesteps x channels), cast to float
    features_raw = df[feat_cols].values.astype(np.float32)

    # Resample each channel independently
    src_hz = _estimate_source_hz(df)
    logger.debug("%s: estimated %.1f Hz -> resampling to %d Hz", filename, src_hz, TARGET_HZ)

    # Resample first channel to get actual output length
    ch0_resampled = _resample_array(features_raw[:, 0], src_hz, TARGET_HZ)
    actual_len = len(ch0_resampled)

    # Resample remaining channels with length normalization
    features_rs = np.zeros((actual_len, n_channels), dtype=np.float32)
    features_rs[:, 0] = ch0_resampled
    for ch in range(1, n_channels):
        ch_resampled = _resample_array(features_raw[:, ch], src_hz, TARGET_HZ)
        # Handle minor length discrepancies (truncate or pad)
        if len(ch_resampled) > actual_len:
            features_rs[:, ch] = ch_resampled[:actual_len]
        elif len(ch_resampled) < actual_len:
            features_rs[:len(ch_resampled), ch] = ch_resampled
        else:
            features_rs[:, ch] = ch_resampled

    # Resample labels to match actual feature length
    label_indices = np.round(
        np.linspace(0, len(labels) - 1, actual_len)
    ).astype(int)
    labels_rs = labels[label_indices]

    # Zero-pad if shorter than one window
    if actual_len < WINDOW_STEPS:
        pad_len = WINDOW_STEPS - actual_len
        logger.warning(
            "%s: only %d steps after resampling; zero-padding %d steps.",
            filename, actual_len, pad_len
        )
        # Pad with repeated last value for each channel (more realistic than zeros)
        last_values = features_rs[-1:, :]  # (1, n_channels)
        pad_features = np.tile(last_values, (pad_len, 1))
        features_rs = np.vstack([features_rs, pad_features])

        # Pad labels with the dominant label in the file
        valid_labels = labels_rs[labels_rs >= 0]
        if len(valid_labels) > 0:
            dominant_label = int(np.bincount(valid_labels).argmax())
        else:
            dominant_label = 0  # Fallback to label 0 if no valid labels
        labels_rs = np.pad(labels_rs, (0, pad_len), constant_values=dominant_label)

    # Choose stride depending on mode (train or test)
    stride = STRIDE_STEPS if mode == "train" else TEST_STRIDE_STEPS
    X, y = _extract_windows(features_rs, labels_rs, WINDOW_STEPS, stride, mode=mode)
    logger.info("%s: %d windows extracted (src %.1f Hz)", filename, len(X), src_hz)
    return X, y
