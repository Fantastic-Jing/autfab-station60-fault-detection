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
from datetime import datetime

from config import OUTPUT_DIR, LABEL_STRATEGY
from dataset import build_dataset
from train_eval import run_experiments




def setup_logging(run_dir: str, log_level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    # Log to both console and file simultaneously
    logging.basicConfig(level=log_level, format=fmt, datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)


def save_summary(all_results: list[dict], run_dir: str) -> None:
    """Write a human-readable text summary of all metric results."""
    out_path = os.path.join(run_dir, "results_summary.txt")

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

def save_config_snapshot(run_dir: str) -> None:
    """Save all current config parameters to a text file for reproducibility."""
    import config as cfg
    out_path = os.path.join(run_dir, "config_snapshot.txt")
    lines = ["Config Snapshot", "=" * 40, ""]
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        val = getattr(cfg, key)
        if callable(val):
            continue
        lines.append(f"{key} = {val!r}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info("Config snapshot saved to %s", out_path)


def main() -> None:
    # Create timestamped run directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    setup_logging(run_dir)
    logger = logging.getLogger(__name__)

    logger.info("Using label strategy: %s", LABEL_STRATEGY)

    # build_dataset
    logger.info("Building dataset with label strategy '%s' ...", LABEL_STRATEGY)
    dataset = build_dataset(label_strategy=LABEL_STRATEGY)

    logger.info(
        "Train samples: %d  |  Test samples: %d",
        len(dataset["y_train"]), len(dataset["y_test"])
    )

    logger.info("Running ...")
    all_results = run_experiments(dataset, run_dir)

    save_config_snapshot(run_dir)
    logger.info("Done. Config saved to '%s/'.", run_dir)

    save_summary(all_results, run_dir)
    logger.info("Done. Results saved to '%s/'.", run_dir)


if __name__ == "__main__":
    main()
