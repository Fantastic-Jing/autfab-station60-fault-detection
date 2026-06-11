"""
Entry point for the Station 60 fault detection pipeline.

Usage:
    python main.py

Outputs (written to OUTPUT_DIR):
    - cm_MiniROCKET.png
    - cm_DTW-1NN.png
    - model_comparison.png
    - results_summary.txt
"""

import logging
import os
import json
import numpy as np

from config import OUTPUT_DIR
from dataset import build_dataset
from train_eval import run_experiments


def setup_logging(log_level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(level=log_level, format=fmt, datefmt="%H:%M:%S")


def save_summary(all_results: list[dict], save_dir: str = OUTPUT_DIR) -> None:
    """Write a human-readable text summary of all metric results."""
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "results_summary.txt")

    lines = ["Station 60 Fault Detection — Results Summary", "=" * 60, ""]
    for r in all_results:
        lines.append(f"Model: {r['clf_name']}")
        lines.append(f"  Accuracy       : {r['accuracy']:.4f}")
        lines.append(f"  Macro F1       : {r['f1_macro']:.4f}")
        lines.append(f"  Weighted F1    : {r['f1_weighted']:.4f}")
        lines.append(f"  Train time     : {r['train_time_s']:.2f} s")
        lines.append(f"  Infer time     : {r['infer_time_s']:.3f} s total")
        lines.append(f"  Infer per sample: {r['infer_ms_per_sample']:.3f} ms")
        lines.append("  Per-class F1:")
        for label, f1 in r["per_class_f1"].items():
            lines.append(f"    Label {label:>2d}: {f1:.4f}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    logging.getLogger(__name__).info("Summary written to %s", out_path)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Building dataset ...")
    dataset = build_dataset()

    # logger.info(
    #     "Train samples: %d  |  Test samples: %d",
    #     len(dataset["y_train"]), len(dataset["y_test"])
    # )

    # logger.info("Running experiments ...")
    # all_results = run_experiments(dataset)
    #
    # save_summary(all_results)
    # logger.info("Done. Results saved to '%s/'.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
