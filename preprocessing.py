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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    Resample a 1-D signal from src_hz to dst_hz using polyphase filtering.
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the time axis.
    Pure Window strategy: only keep windows where every timestep shares the same label.

    Returns:
        X: (n_windows, n_channels, window_len)
        y: (n_windows,)
    """
    n_timesteps, n_channels = features.shape
    windows_X, windows_y = [], []

    for start in range(0, n_timesteps - window_len + 1, stride):
        end = start + window_len
        window_labels = labels[start:end]

        unique = np.unique(window_labels)
        if len(unique) != 1:
            continue                  # mixed-label window; skip

        label = int(unique[0])
        if label not in VALID_LABELS:
            continue                  # discard unmapped / out-of-range labels

        windows_X.append(features[start:end].T)   # (n_channels, window_len)
        windows_y.append(label)

    if len(windows_X) == 0:
        return np.empty((0, n_channels, window_len), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(windows_X).astype(np.float32), np.array(windows_y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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


def process_file(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing for a single CSV file.

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

    n_resampled = int(round(len(features_raw) * TARGET_HZ / src_hz))
    features_rs = np.zeros((n_resampled, n_channels), dtype=np.float32)
    for ch in range(n_channels):
        features_rs[:, ch] = _resample_array(features_raw[:, ch], src_hz, TARGET_HZ)

    # Resample labels using nearest-neighbour (no interpolation for discrete values)
    label_indices = np.round(
        np.linspace(0, len(labels) - 1, n_resampled)
    ).astype(int)
    labels_rs = labels[label_indices]

    # Zero-pad if shorter than one window
    if n_resampled < WINDOW_STEPS:
        pad_len = WINDOW_STEPS - n_resampled
        logger.warning(
            "%s: only %d steps after resampling; zero-padding %d steps.",
            filename, n_resampled, pad_len
        )
        features_rs = np.pad(features_rs, ((0, pad_len), (0, 0)), mode="constant")
        # Pad labels with the dominant label in the file
        dominant_label = int(np.bincount(labels_rs[labels_rs >= 0]).argmax())
        labels_rs = np.pad(labels_rs, (0, pad_len), constant_values=dominant_label)

    X, y = _extract_windows(features_rs, labels_rs, WINDOW_STEPS, STRIDE_STEPS)
    logger.info("%s: %d windows extracted (src %.1f Hz)", filename, len(X), src_hz)
    return X, y
