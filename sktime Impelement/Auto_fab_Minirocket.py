import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets._readers_writers.ts import load_from_tsfile_to_dataframe
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable


def load_ts_file(path: Path):
    """Load a .ts file into X and y."""
    X, y = load_from_tsfile_to_dataframe(str(path), return_separate_X_and_y=True)
    return X, y


def main() -> None:
    root = Path(__file__).resolve().parent
    train_path = root / "TrainV3.ts"
    test_path = root / "TestV3.ts"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train.ts and Test.ts must be present in the same folder as this script."
        )

    X_train, y_train = load_ts_file(train_path)
    X_test, y_test = load_ts_file(test_path)

    print("=" * 70)
    print("Dataset: AutoFab general information")
    print("=" * 70)
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    print("Number of dimensions:", X_train.shape[1])
    print("Labels:", np.unique(y_train))

    seed = 42
    np.random.seed(seed)

    pipeline = make_pipeline(
        MiniRocketMultivariateVariable(
            pad_value_short_series=-10.0,
            random_state=seed,
            max_dilations_per_kernel=16,
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

    metrics = pd.DataFrame(
        [
            [
                "AutoFab_TrainTest",
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

    print("\nAll predictions:")
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        print(f"Sample {i}: Predicted = {pred}, Actual = {true}")

    print("\nMisclassified samples:")
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        if pred != true:
            print(f"Sample {i}: Predicted = {pred}, Actual = {true}")

    # print("\nSample structure and lengths:")
    # sample_idx = 0
    # sample = X_train.iloc[sample_idx]
    # print("=" * 70)
    # print(f"DATA STRUCTURE SAMPLE {sample_idx}")
    # print("=" * 70)
    # print(f"Label: {y_train[sample_idx]}")
    # print(f"Type: {type(sample)}")
    # print(f"Number of dimensions: {len(sample)}")
    # print(f"Dimension names: {sample.index.tolist()}")

    # print("\nLength of each dimension for the sample:")
    # for dim in sample.index:
    #     series = sample[dim]
    #     print(f"{dim}: {len(series)}")

    print("\nLengths for each training sample")
    for i in range(len(y_train)):
        lengths = [len(X_train.iloc[i, j]) for j in range(X_train.shape[1])]
        if len(set(lengths)) == 1:
            print(f"Sample {i}: length = {lengths[0]}")
        else:
            print(f"Sample {i}: lengths = {lengths}")


if __name__ == "__main__":
    main()
