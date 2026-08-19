# Station 60 Fault Detection

Real-time fault classification for the AutFab smart factory Station 60 pneumatic press — 51-channel multivariate sensor time series classified into 15 fault types using **MiniROCKET** via the [aeon](https://www.aeon-toolkit.org/) toolkit.

**Tools:** Python 3.10+, aeon ≥ 0.10, numpy, scipy, pandas, scikit-learn, numba, matplotlib  
**Platform:** Laboratory PC, CPU only (no GPU required)

---

## The Problem

Station 60 is a pneumatic press in the AutFab smart factory at Hochschule Darmstadt. When a fault occurs, operators need to identify the fault type quickly to get the station back online. The station produces 15 distinct fault types, many of which generate sensor patterns that look similar to each other or to normal operation. Manual diagnosis under time pressure is slow and error-prone.

The goal is a model that reads a short window of sensor data and outputs the fault class in under 1 ms — fast enough for real-time deployment on a factory floor.

---

## Dataset

| | |
|---|---|
| Source | AutFab smart factory, Station 60, Hochschule Darmstadt |
| Recordings | 64 CSV files, one per fault scenario |
| Sensor channels | 51 |
| Raw sampling rate | 56–79 Hz (varies per file) |
| Total rows | ~114,000 |
| Classification target | 15 classes (Label 0–14) |

**Due to data privacy, the raw CSV files are not included in this repository.**

### Fault classes

| Label | Fault type |
|---|---|
| 0 | Normal behaviour |
| 1 | Station not safe *(includes labels 15, 16, 17 — see note below)* |
| 2 | Area stop |
| 3 | Light curtain |
| 4 | Pressure mat |
| 5 | Door safety switch |
| 6 | Emergency stop |
| 7 | Safety components passivated |
| 8 | Position sensor of press broken |
| 9 | RFID presence defunct |
| 10 | Shuttle-in-station sensor defunct |
| 11 | No connection to ET200S |
| 12 | No connection to control station |
| 13 | No connection to PROFINET |
| 14 | No connection to RFID |

> **Note on labels 15, 16, 17:** These states (station not in auto mode / press not at start position / software error) only appear in fault recovery segments. No dedicated recordings exist for them. They are remapped to Label 1 during preprocessing.

---

## Pipeline

### 1. Preprocessing

**The problem:** Each CSV file has a slightly different sampling rate (56–79 Hz). Labels switch multiple times within a single file (normal → fault active → recovery). Some files are shorter than one window.

**The implementation:**

- Estimate source frequency from the nanosecond timestamp column using median inter-sample interval
- Resample every channel to **64 Hz** using a polyphase filter with exact rational ratio (via `scipy.signal.resample_poly`)
- Remap labels 15, 16, 17 → 1
- Sync the label array to the resampled length using nearest-neighbour interpolation
- Zero-pad files shorter than one window using the last known sensor value

### 2. Windowing Strategy

**The problem:** A naive sliding window produces *mixed windows* that span label boundaries. For short fault segments (e.g. Pressure Mat in the test file: ~40 rows), a 128-step window produces zero valid samples. The test set also needs to mirror real deployment, where sensor data is a continuous unlabeled stream and no windows can be discarded.

**The implementation:**

Training set uses two passes:

- **Pass 1 — Pure Window** (stride = 64): keep only windows where every timestep shares one label. Guarantees clean, noise-free training samples.
- **Pass 2 — First Fault Label** (stride = 16): keep all windows. Assign the first non-zero label found inside the window. Exposes the model to fault onset patterns. Downsampled to **4:1 ratio** vs Pass 1 to avoid dominating training.

Test set uses First Fault Label only (stride = 16, no windows discarded).

**Per-class stride** for minority classes:

| Class | Stride |
|---|---|
| 2 Area Stop | 2 |
| 3 Light Curtain | 1 |
| 4 Pressure Mat | 1 |
| 5 Door Safety Switch | 1 |
| 6 Emergency Stop | 1 |
| 9 RFID Presence Defunct | 4 |
| 11 No Connection ET200S | 8 |
| 12 No Connection Control Station | 4 |
| all others | 64 (default) |

### 3. Class Balancing

**The problem:** Raw window counts vary drastically across classes — Normal and Transition produce thousands of windows while Pressure Mat and Area Stop produce fewer than 50. A biased training set pushes the model toward majority classes.

**The implementation — three steps applied in order:**

1. **Truncate:** cap each class at 400 windows (train) / 200 windows (test) by random downsampling
2. **Augment:** top up classes below 200 samples by copying windows with small Gaussian noise (std = 0.02). Applied to both train and test for a fair evaluation
3. **Normalise:** per-channel `StandardScaler` fitted on the training set only, then applied to test. Channels with std < 1e-7 are dropped before training to avoid MiniROCKET's variance check

Final dataset: **6000 training samples · 3000 test samples · 200 per class · 15 classes**

### 4. Training & Evaluation

**MiniROCKET** (`aeon.classification.convolution_based.MiniRocketClassifier`):
- 10,000 random convolutional kernels extract PPV features from each channel
- A Ridge classifier is trained on top
- JIT-compiled via Numba — no GPU needed

Evaluation metrics: Accuracy, Macro F1, Weighted F1, per-class F1, confusion matrix, training time, inference time per sample.

---

## Results

### Final performance

| Metric | Value |
|---|---|
| Accuracy | **98.1%** |
| Macro F1 | **0.981** |
| Weighted F1 | **0.981** |
| Training time | 32 s |
| Inference / sample | **0.365 ms** |
| Test samples | 3000 (200 per class) |

Confusion matrix: [`results/2026-07-10_23-01-24/cm_MiniROCKET.png`](./results/2026-07-10_23-01-24/cm_MiniROCKET.png)

*Figure 1: Final confusion matrix — class indices 0–14 match the fault class table above.*

### Optimisation history

The pipeline went through several iterations before reaching the final result:

| Stage | Key change | Accuracy | Macro F1 |
|---|---|---|---|
| Baseline | Pure Window both sides, stride = 64 | 0.99 | 0.93 |
| Step 1 | Test set → First Fault Label, stride = 16 | 0.92 | 0.81 |
| Step 2 | + Train FFL mixed windows, 1:1 ratio | 0.85 | 0.85 |
| Step 3 | + Train FFL mixed windows, 4:1 ratio | 0.92 | 0.88 |
| Step 4 | Window 128 → 64 steps, per-class stride | 0.98 | 0.98 |
| **Final** | **+ augmentation (noise std = 0.02)** | **0.981** | **0.981** |

The baseline 0.99 accuracy looks better but is misleading — it used Pure Window on the test set, which discarded all boundary windows and produced zero test samples for Pressure Mat (Case 4). Steps 1–3 improved fairness at the cost of raw numbers. Steps 4 and Final recovered accuracy while keeping the evaluation honest.

---

## Limitations

**Pressure Mat (Case 4) confusion with Transition (Case 1):** A small number of Case 4 test windows are still misclassified as Transition. The root cause is that the original recording for Case 4 in the test file contains only ~40 rows of fault signal. Even with stride = 1, the mixed windows that can be extracted contain a high proportion of Transition signal. This is a data collection issue, not a model limitation — longer recordings of this fault type would resolve it.

**Augmented test samples:** The test set uses the same augmentation pipeline as the training set to ensure every class reaches 200 samples for a fair evaluation. This means some test samples are not real recordings. Real-world deployment performance should be validated against a freshly collected held-out set.

**First Fault Label boundary noise:** Assigning the first non-zero label to a mixed window is a heuristic. In windows that contain a very short fault onset followed by a long recovery segment, the label reflects the fault type but the signal content is dominated by the recovery pattern. This is an inherent trade-off of the labeling strategy.

---

## How to Run

```bash
pip install -r requirements.txt

# Place all CSV files in data/raw/
python main.py
```

Each run creates a timestamped folder under `results/` containing:

| File | Content |
|---|---|
| `run.log` | Full run log |
| `summary.txt` | Metric summary |
| `cm_MiniROCKET.png` | Confusion matrix |
| `config_snapshot.txt` | All config parameters at run time |

To change parameters (window size, stride, augmentation noise, etc.), edit `config.py`. No other files need to be touched.

---

## Files

| File | Description |
|---|---|
| `config.py` | All tunable parameters in one place |
| `preprocessing.py` | CSV loading, resampling to 64 Hz, sliding window extraction |
| `dataset.py` | File-level train/test split, normalisation, class balancing, augmentation |
| `train_eval.py` | MiniROCKET training, evaluation, confusion matrix and comparison plots |
| `main.py` | Entry point — creates timestamped output folder, saves config snapshot |
| `requirements.txt` | Python dependencies |
| `data/file_info_extraction.py` | Extracts row counts and file metadata from all CSV files |
| `data/label_extraction.py` | Extracts label change intervals and segment statistics per file |
| `data/time_stamp_extraction.py` | Estimates per-file sampling frequency from nanosecond timestamps |

---

## Acknowledgments

This project was developed as part of the **Master Team Projects** course in the **Electrical Engineering (M.Sc.) — Major Automation** programme at **Hochschule Darmstadt (h_da)**, Faculty of Electrical Engineering and Information Technology.

**Course:** Master Team Projects SoSe 2026  
**Supervised by:**
- Prof. Dr. Stephan Simons (Responsible for Lab)
- Heiko Webert, M.Sc. (Laboratory Engineer)

**Team:** Jing Wen · Hoang Viet Tung

The sensor data used in this project was recorded from the AutFab smart factory at Hochschule Darmstadt, Station 60 pneumatic press. Due to data privacy, the raw dataset is not included in this repository.