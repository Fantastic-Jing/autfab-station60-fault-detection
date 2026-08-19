"""
Model training and evaluation.

Two classifiers are trained and compared:
  - MiniRocketClassifier  (aeon)
  - KNeighborsTimeSeriesClassifier with DTW distance  (aeon)

Metrics reported per model:
  accuracy, macro F1, weighted F1, per-class F1,
  training time, inference time, confusion matrix.
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

from config import LABEL_NAMES, OUTPUT_DIR, RANDOM_SEED, MINIROCKET_NUM_KERNELS, DTW_N_NEIGHBORS, DTW_WINDOW
import os

logger = logging.getLogger(__name__)



# Classifier factory


def build_minirocket(n_jobs: int = -1) -> MiniRocketClassifier:
    return MiniRocketClassifier(
        n_kernels=MINIROCKET_NUM_KERNELS,
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
    )


def build_dtw_knn(n_neighbors: int = DTW_N_NEIGHBORS, window: float = DTW_WINDOW) -> KNeighborsTimeSeriesClassifier:
    """
    1-NN with DTW distance and a Sakoe-Chiba band (window fraction of series length).
    n_jobs is not supported for DTW-kNN in aeon; inference is inherently sequential.
    """
    return KNeighborsTimeSeriesClassifier(
        n_neighbors=n_neighbors,
        distance="dtw",
        distance_params={"window": window},
    )



# Train + evaluate
def _format_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> str:
    """Print confusion matrix with numeric indices for readability."""
    n = len(class_names)
    col_w = 6

    # Header: 0, 1, 2 ... as column indices
    header_nums = " ".join(f"{i:>{col_w}}" for i in range(n))
    row_label_w = max(len(n_) for n_ in class_names)
    header = " " * (row_label_w + 2) + header_nums
    sep = " " * (row_label_w + 2) + "-" * (n * (col_w + 1) - 1)

    # Each row: full class name + index + counts
    rows = [header, sep]
    for i, row in enumerate(cm):
        label = f"{class_names[i]:>{row_label_w}}"
        values = " ".join(f"{v:>{col_w}}" for v in row)
        rows.append(f"{label}  {values}")

    # Legend: index -> class name
    rows.append("")
    rows.append("Predicted class index legend:")
    for i, name in enumerate(class_names):
        rows.append(f"  {i:>2}: {name}")

    return "\n".join(rows)


def train_and_evaluate(
    clf,
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: dict = LABEL_NAMES,
) -> dict:
    """
    Fit clf on training data, predict on test data, compute and log all metrics.

    Returns a dict with all numeric results for downstream comparison.
    """
    present_labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    class_names = [label_names.get(l, str(l)) for l in present_labels]

    # Training
    logger.info("[%s] Starting training on %d samples ...", clf_name, len(X_train))
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    logger.info("[%s] Training done in %.2f s", clf_name, train_time)

    # Inference
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = time.perf_counter() - t0
    infer_time_per_sample = infer_time / len(X_test) * 1000   # ms per sample
    logger.info(
        "[%s] Inference on %d samples: %.3f s total / %.3f ms per sample",
        clf_name, len(X_test), infer_time, infer_time_per_sample,
    )

    # Metrics
    acc          = accuracy_score(y_test, y_pred)
    f1_macro     = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=present_labels, zero_division=0)
    cm           = confusion_matrix(y_test, y_pred, labels=present_labels)

    logger.info("[%s] Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
                clf_name, acc, f1_macro, f1_weighted)
    logger.info("[%s] Full classification report:\n%s", clf_name,
                classification_report(y_test, y_pred, labels=present_labels,
                                      target_names=class_names, zero_division=0))
    logger.info("[%s] Confusion matrix:\n%s", clf_name,
                _format_confusion_matrix(cm, class_names))

    return {
        "clf_name":              clf_name,
        "accuracy":              acc,
        "f1_macro":              f1_macro,
        "f1_weighted":           f1_weighted,
        "per_class_f1":          dict(zip(present_labels, per_class_f1.tolist())),
        "train_time_s":          train_time,
        "infer_time_s":          infer_time,
        "infer_ms_per_sample":   infer_time_per_sample,
        "confusion_matrix":      cm,
        "present_labels":        present_labels,
        "class_names":           class_names,
    }


# Visualisation helpers
def plot_confusion_matrix(results: dict, save_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    cm           = results["confusion_matrix"]
    class_names  = results["class_names"]
    n            = len(class_names)

    # Use numeric indices as tick labels
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(n)),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix — {results['clf_name']}")

    # Add legend below the plot
    legend_lines = [f"{i}: {name}" for i, name in enumerate(class_names)]
    legend_text  = "   ".join(legend_lines[:8]) + "\n" + "   ".join(legend_lines[8:])
    fig.text(0.01, 0.01, legend_text, fontsize=7, verticalalignment="bottom",
             family="monospace")

    fig.tight_layout(rect=[0, 0.08, 1, 1])   # leave space at bottom for legend
    out_path = os.path.join(save_dir, f"cm_{results['clf_name'].replace(' ', '_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)


def plot_metric_comparison(all_results: list[dict], run_dir: str) -> None:
    """Bar chart comparing key metrics across all evaluated classifiers."""
    os.makedirs(run_dir, exist_ok=True)

    names        = [r["clf_name"]      for r in all_results]
    accuracy     = [r["accuracy"]      for r in all_results]
    f1_macro     = [r["f1_macro"]      for r in all_results]
    f1_weighted  = [r["f1_weighted"]   for r in all_results]
    train_times  = [r["train_time_s"]  for r in all_results]
    infer_ms     = [r["infer_ms_per_sample"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(names))
    bar_w = 0.25

    # Accuracy + F1
    ax = axes[0]
    ax.bar(x - bar_w, accuracy,    bar_w, label="Accuracy")
    ax.bar(x,         f1_macro,    bar_w, label="Macro F1")
    ax.bar(x + bar_w, f1_weighted, bar_w, label="Weighted F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_title("Classification Metrics")
    ax.legend()

    # Training time
    ax = axes[1]
    ax.bar(x, train_times, color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_title("Training Time (s)"); ax.set_ylabel("seconds")

    # Inference time per sample
    ax = axes[2]
    ax.bar(x, infer_ms, color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_title("Inference Time per Sample (ms)"); ax.set_ylabel("ms")

    fig.suptitle("Model Comparison — Station 60 Fault Detection")
    fig.tight_layout()

    out_path = os.path.join(run_dir, "model_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison chart saved to %s", out_path)



# Main entry point


def run_experiments(dataset: dict, run_dir: str) -> list[dict]:
    """
    Run MiniROCKET and DTW+kNN on the provided dataset dict.
    
    Args:
        dataset: dict with X_train, y_train, X_test, y_test
        run_dir: directory to save outputs
    
    Returns a list of result dicts (one per classifier).
    """
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]

    classifiers = [
        (build_minirocket(),    "MiniROCKET"),
        # (build_dtw_knn(),       "DTW-1NN"),
    ]

    all_results = []
    for clf, name in classifiers:
        results = train_and_evaluate(clf, name, X_train, y_train, X_test, y_test)
        plot_confusion_matrix(results, run_dir)
        all_results.append(results)

    # plot_metric_comparison(all_results, run_dir)
    return all_results
