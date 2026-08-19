# Station 60 Fault Detection

A machine learning pipeline for reactive fault detection on the **AutFab smart factory Station 60 pneumatic press**. When a fault occurs, the system classifies the fault type from multivariate sensor time series, helping operators identify the problem quickly.

## Implementations

The two independent project implementations are kept in separate modules:

- `modules/Aeon-Implementation/` — the Aeon-based implementation
- `modules/Sktime-Implement/` — the sktime-based implementation

Each module has its own source files, dependencies, data utilities, and results. Run commands from the relevant module directory.


---

## What it does

Station 60 has 14 distinct fault types (plus normal operation), each producing different patterns across 44 sensor channels. This pipeline:

1. Loads raw sensor recordings (CSV) from the station
2. Resamples all files to a unified 64 Hz
3. Slices the time series into 2-second windows
4. Trains two classifiers and compares their performance

---

## Results (first run)

| Metric | MiniROCKET | DTW-1NN |
|---|---|---|
| Accuracy | 98.55% | 98.18% |
| Macro F1 | 0.993 | 0.906 |
| Weighted F1 | 0.986 | 0.985 |
| Train time | 17.9 s | 0.06 s |
| Inference / sample | **0.72 ms** | 48.2 ms |

MiniROCKET is the clear winner for production use — 67× faster inference with higher F1. DTW-1NN serves as an interpretable baseline.

> **Known issue:** Label 4 (Pressure Mat) scores F1 = 0.00 for both models. This is a data issue — the fault segments in the training files are too short to produce enough pure windows. Under investigation.

---

## Dataset

- 64 CSV recordings, one per fault scenario
- 44 sensor channels (boolean I/O signals, pressure sensors, position sensors, RFID, diagnostics)
- Sampling rate: 56–79 Hz (varies per file, unified to 64 Hz in preprocessing)
- ~114,000 rows total

| Label | Fault Type |
|---|---|
| 0 | Normal operation |
| 1 | Transition / station not safe |
| 2 | Area stop |
| 3 | Light curtain |
| 4 | Pressure mat |
| 5 | Door safety switch |
| 6 | Emergency stop |
| 7 | Safety components passivated (QBad) |
| 8 | Position sensor broken/blocked |
| 9 | RFID presence defunct |
| 10 | Shuttle-in-station sensor defunct |
| 11 | No connection to ET200S |
| 12 | No connection to control station |
| 13 | No connection to PROFINET |
| 14 | No connection to RFID |

Labels 15, 16, 17 (intermediate recovery states) are remapped to Label 1 during preprocessing.

---

## Project structure

```
project/
├── config.py          # all parameters in one place (paths, window size, label mapping)
├── preprocessing.py   # CSV loading, resampling, sliding window extraction
├── dataset.py         # file-level train/test split, per-channel normalization
├── train_eval.py      # model training, evaluation, plots
├── main.py            # entry point
└── results/           # output metrics, confusion matrices, comparison charts
```

---

## How to run

**Install dependencies:**
```bash
pip install aeon numpy scipy pandas numba scikit-learn matplotlib
```

**Place all CSV files in** `data/raw/`

**Run:**
```bash
python main.py
```

Outputs are written to `results/`:
- `results_summary.txt` — metric comparison table
- `cm_MiniROCKET.png` — confusion matrix
- `cm_DTW-1NN.png` — confusion matrix
- `model_comparison.png` — side-by-side bar chart
- `run.log` — full run log

---

## Key design decisions

**File-level train/test split**
The test set is always the last recorded file per fault class. Splitting by file (not by row) prevents data leakage — adjacent windows from the same recording are highly correlated.

**Pure Window labeling**
Each recording contains multiple label phases (normal → fault active → recovery). A window is only kept if every timestep inside it shares the same label. This avoids feeding the model ambiguous boundary windows.

**Unified resampling to 64 Hz**
Raw files have slightly different sampling rates (56–79 Hz). Before windowing, all channels are resampled using polyphase filtering so that 128 steps always equals exactly 2 seconds across every file.

**Per-channel normalization**
Sensor channels have very different scales (boolean 0/1 vs. pressure values in bar). A standard scaler is fitted on the training set only and applied to both splits.

---

## Requirements

- Python 3.10+
- aeon ≥ 0.10
- numpy, scipy, pandas, scikit-learn, matplotlib
- numba ≥ 0.55 (required for MiniROCKET)

---

## About this project

This project was developed as part of the **Master Team Projects** course in the  
**Electrical Engineering (M.Sc.) — Major Automation** program at  
**Hochschule Darmstadt (h_da), Faculty of Electrical Engineering and Information Technology**.

**Course:** Master Team Projects SoSe 2026  
**Supervised by:**
- Prof. Dr. Stephan Simons (Responsible for Lab)
- Heiko Webert, M.Sc. (Laboratory Engineer)

**Team:**
- Jing Wen
- Hoang Viet Tung

The sensor data used in this project was recorded from the **AutFab smart factory at Hochschule Darmstadt**, Station 60 pneumatic press.  
Due to data privacy, the raw dataset is not included in this repository.
