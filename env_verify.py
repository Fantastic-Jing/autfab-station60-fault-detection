import time
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.convolution_based import MiniRocketClassifier


def generate_mock_sensor_data():
    """
    Simulates multi-channel time-series data from AutFab Station 60.
    Format required by aeon: (n_cases, n_channels, n_timepoints)
    """
    print(">>> Generating mock time-series data for Station 60...")

    # 20 samples, 3 sensor channels (e.g., pressure, airflow, vibration), 150 time steps
    X_train = np.random.rand(20, 3, 150)
    y_train = np.array([
        'Normal', 'Leakage_Fault', 'Normal', 'Pressure_Drop', 'Leakage_Fault',
        'Normal', 'Pressure_Drop', 'Leakage_Fault', 'Normal', 'Pressure_Drop',
        'Normal', 'Leakage_Fault', 'Normal', 'Pressure_Drop', 'Leakage_Fault',
        'Normal', 'Pressure_Drop', 'Leakage_Fault', 'Normal', 'Pressure_Drop'
    ])

    # Test data (5 samples)
    X_test = np.random.rand(5, 3, 150)
    y_test = np.array(['Normal', 'Leakage_Fault', 'Pressure_Drop', 'Normal', 'Pressure_Drop'])

    return X_train, y_train, X_test, y_test


def evaluate_classifier(clf, name, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a classifier, tracking execution time and accuracy.
    """
    print(f"\n================ [{name} Evaluation] ================")

    # Measure Training Time
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"[{name}] Training completed in {train_time:.4f} seconds.")

    # Measure Inference Time
    start_time = time.time()
    predictions = clf.predict(X_test)
    inference_time = time.time() - start_time
    print(f"[{name}] Inference completed in {inference_time:.4f} seconds.")

    # Calculate Accuracy
    accuracy = np.mean(predictions == y_test)

    return {
        "Classifier": name,
        "Train Time (s)": round(train_time, 4),
        "Inference Time (s)": round(inference_time, 4),
        "Accuracy (Mock)": f"{accuracy * 100:.1f}%",
        "Predictions": list(predictions)
    }


if __name__ == "__main__":
    print("=== AutFab Station 60 Fault Detection Environment Test ===")

    # 1. Prepare Data
    X_train, y_train, X_test, y_test = generate_mock_sensor_data()

    # 2. Initialize Models
    # Pipeline 1: DTW + kNN (Distance-based approach)
    knn_dtw = KNeighborsTimeSeriesClassifier(distance="dtw", n_neighbors=3)

    # Pipeline 2: MiniROCKET (Convolution-based approach)
    minirocket = MiniRocketClassifier()

    # 3. Run Evaluation
    results = []
    results.append(evaluate_classifier(knn_dtw, "DTW+kNN", X_train, y_train, X_test, y_test))
    results.append(evaluate_classifier(minirocket, "MiniROCKET", X_train, y_train, X_test, y_test))

    # 4. Print Summary Report
    print("\n================ [Final Benchmark Summary] ================")
    print(f"{'Classifier':<15} | {'Train Time':<12} | {'Inference Time':<15} | {'Mock Accuracy':<15}")
    print("-" * 65)
    for res in results:
        print(
            f"{res['Classifier']:<15} | {res['Train Time (s)']:<12} | {res['Inference Time (s)']:<15} | {res['Accuracy (Mock)']:<15}")
    print("-" * 65)

    print("\n>>> Environment verification successful! All algorithms are fully functional.")