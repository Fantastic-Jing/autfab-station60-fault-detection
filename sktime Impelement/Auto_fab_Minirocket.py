from matplotlib.pylab import seed
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets._readers_writers.ts import load_from_tsfile_to_dataframe
#from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from sklearn.utils import resample
from collections import Counter

def load_ts_file(path: Path):
    """Load a .ts file into X and y."""
    X, y = load_from_tsfile_to_dataframe(str(path), return_separate_X_and_y=True)
    return X, y


def normalize_per_dimension_per_sample(X):
    """
    Normalize từng time series trong mỗi cell (sample, dimension)
    bằng min-max normalization vào [0, 1], TRỪ dimension cuối (label).
    """
    X_new = X.copy()
    n_samples, n_dims = X.shape

    for i in range(n_samples):
        for j in range(n_dims - 1):
            series = X_new.iloc[i, j]
            arr = np.asarray(series, dtype=float)

            if arr.size == 0:
                continue

            min_val = arr.min()
            max_val = arr.max()

            if max_val - min_val == 0:
                arr_norm = np.zeros_like(arr, dtype=float)
            else:
                arr_norm = (arr - min_val) / (max_val - min_val)

            X_new.iloc[i, j] = pd.Series(arr_norm)

    return X_new


def window_sliding_augmentation_from_last_dim(X, window_size, stride=1):
    """
    Tạo sample mới bằng window sliding với độ dài window ngẫu nhiên.
    - Feature: tất cả dimension trừ dimension cuối.
    - Label của window: mode của dimension cuối trong window.

    Ghi chú:
    - Biến biên độ (amplitude) là biến cục bộ trong hàm.
    - Ví dụ window_size=100, amplitude=10 -> window length ngẫu nhiên trong [90, 110].
    """
    X_list = []
    y_list = []

    n_samples, n_dims = X.shape
    dim_names = X.columns.tolist()

    feature_dim_names = dim_names[:-1]
    label_dim_name = dim_names[-1]

    amplitude = 0  # chỉnh tay mỗi lần chạy nếu muốn

    min_window_size = max(1, window_size - amplitude)
    max_window_size = window_size + amplitude

    for i in range(n_samples):
        sample = X.iloc[i]
        label_series = sample[label_dim_name]
        label_arr = np.asarray(label_series)

        T = len(label_arr)

        # Nếu sample còn ngắn hơn cả window nhỏ nhất thì bỏ qua
        if T < min_window_size:
            continue

        for start in range(0, T - min_window_size + 1, stride):
            # Lấy ngẫu nhiên độ dài window cho lần cắt này
            current_window_size = np.random.randint(
                min_window_size, max_window_size + 1
            )

            end = start + current_window_size

            # Nếu window vượt quá chiều dài sample thì bỏ qua
            if end > T:
                continue

            window_labels = label_arr[start:end]
            #window_label = pd.Series(window_labels).value_counts().idxmax()
            window_label = window_labels[-1]

            new_row = {}
            for dim_name in feature_dim_names:
                series = sample[dim_name]
                arr = np.asarray(series)
                window_feat = arr[start:end]
                new_row[dim_name] = pd.Series(window_feat)

            X_list.append(new_row)
            y_list.append(window_label)

    X_aug = pd.DataFrame(X_list)
    X_aug = X_aug[feature_dim_names]
    y_aug = np.array(y_list)

    return X_aug, y_aug


def split_windows_train_test(X_aug, y_aug):
    """
    Cứ 3 sample vào train, sample thứ 4 vào test, lặp lại.
    """
    train_rows = []
    train_labels = []
    test_rows = []
    test_labels = []

    for i in range(len(y_aug)):
        if (i + 1) % 4 == 0:
            test_rows.append(X_aug.iloc[i])
            test_labels.append(y_aug[i])
        else:
            train_rows.append(X_aug.iloc[i])
            train_labels.append(y_aug[i])

    X_train = pd.DataFrame(train_rows).reset_index(drop=True)
    X_test = pd.DataFrame(test_rows).reset_index(drop=True)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    return X_train, y_train, X_test, y_test


def print_confusion_matrix_with_labels(y_true, y_pred, labels):
    """
    In confusion matrix lên terminal với label nguyên.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    print("\nConfusion Matrix (Actual vs Predicted):")
    print("(Rows = Actual, Columns = Predicted)")
    print(df_cm.to_string())

def oversample_minority_classes(X, y, random_state=42):
    """
    Oversample các class ít mẫu để số lượng mỗi class bằng class nhiều nhất.
    """
    df = X.copy()
    df["label"] = y

    class_counts = df["label"].value_counts()
    max_count = class_counts.max()

    balanced_parts = []

    for cls in class_counts.index:
        df_cls = df[df["label"] == cls]

        if len(df_cls) < max_count:
            df_cls_resampled = resample(
                df_cls,
                replace=True,
                n_samples=max_count,
                random_state=random_state,
            )
        else:
            df_cls_resampled = df_cls

        balanced_parts.append(df_cls_resampled)

    df_balanced = pd.concat(balanced_parts, axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_balanced = df_balanced["label"].to_numpy()
    X_balanced = df_balanced.drop(columns=["label"])

    return X_balanced, y_balanced

def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "rawV3.ts"

    if not data_path.exists():
        raise FileNotFoundError(
            "rawV3.ts must be present in the same folder as this script."
        )

    # Load only one raw file
    X_raw, _ = load_ts_file(data_path)

    # Normalize all dimensions except the last one (label)
    X_raw = normalize_per_dimension_per_sample(X_raw)

    # Window sliding
    window_size = 80
    stride = 40

    seed = 42
    np.random.seed(seed)

    X_aug, y_aug = window_sliding_augmentation_from_last_dim(
        X_raw, window_size=window_size, stride=stride
    )

    # Balance classes after window sliding beforre split
    # X_aug, y_aug = oversample_minority_classes(X_aug, y_aug, random_state=seed)

    # Split windows: 3 train, 1 test
    X_train, y_train, X_test, y_test = split_windows_train_test(X_aug, y_aug)

    # print("Train class distribution before oversampling:")
    # print(pd.Series(y_train).value_counts().sort_index())

    # Balance classes after window sliding
    X_train, y_train = oversample_minority_classes(X_train, y_train, random_state=seed)
    X_test, y_test = oversample_minority_classes(X_test, y_test, random_state=seed)

    # print("Train class distribution after oversampling:")
    # print(pd.Series(y_train).value_counts().sort_index())

    print("=" * 70)
    print("Dataset: AutoFab window-level from one raw file")
    print("=" * 70)
    print("Total windows:", len(y_aug))
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    print("Number of feature dimensions:", X_train.shape[1])
    print("Labels:", np.unique(np.concatenate([y_train, y_test])))


    pipeline = make_pipeline(
        #MiniRocketMultivariateVariable(
        MiniRocketMultivariate(
            #pad_value_short_series=-10.0,
            random_state=seed,
            max_dilations_per_kernel=16,
            #reference_length="median",
        ),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )

    start = timer()
    pipeline.fit(X_train, y_train)
    end = timer()
    total_training_time = end - start

    start = timer()
    y_pred = pipeline.predict(X_test)
    end = timer()
    inference_time = end - start

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    labels = np.unique(np.concatenate([y_train, y_test]))
    labels = labels.astype(int)

    print_confusion_matrix_with_labels(y_test, y_pred, labels)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.xticks(rotation=45)

    plt.title("Confusion Matrix (window-level)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)

    metrics = pd.DataFrame(
        [
            [
                "AutoFab_OneFile_WindowLevel",
                seed,
                total_training_time,
                inference_time,
                accuracy,
                weighted_f1,
            ]
        ],
        columns=[
            "Dataset",
            "Seed",
            "Total Training Time",
            "Inference Time",
            "Accuracy",
            "Weighted F1",
        ],
    )

    print("\nMetrics:")
    print(metrics)

    print("\nBest alpha selected by RidgeClassifierCV:")
    print(pipeline.named_steps["ridgeclassifiercv"].alpha_)


if __name__ == "__main__":
    main()