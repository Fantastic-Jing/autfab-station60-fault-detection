import numpy as np
import pandas as pd
from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_basic_motions
from sktime.transformations.panel.rocket import MiniRocketMultivariate


# Load Basic Motions dataset
X_train, y_train = load_basic_motions(split="train", return_X_y=True)
X_test, y_test = load_basic_motions(split="test", return_X_y=True)


print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Number of dimensions:", X_train.shape[1])
print("Length of sample 0, dim 0:", X_train.iloc[0, 0].shape[0])
print("Length of sample 1, dim 0:", X_train.iloc[1, 0].shape[0])

print("\nClass labels:")
print("Train:", np.unique(y_train))
print("Test:", np.unique(y_test))


seed = 42
np.random.seed(seed)


stats = []


start = timer()
pipeline = make_pipeline(
    MiniRocketMultivariate(
        random_state=seed,
        max_dilations_per_kernel=16
    ),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)
pipeline.fit(X_train, y_train)
end = timer()


total_training_time = end - start


start = timer()
y_pred = pipeline.predict(X_test)
end = timer()


inference_time = end - start


acc = accuracy_score(y_test, y_pred)
wf1 = f1_score(y_test, y_pred, average="weighted")


stats.append([
    "BasicMotions",
    seed,
    total_training_time,
    inference_time,
    acc,
    wf1
])


stats_df = pd.DataFrame(
    stats,
    columns=[
        "Dataset",
        "Seed",
        "Total Training Time",
        "Inference Time",
        "Accuracy",
        "Weighted F1"
    ]
)


stats_df.to_csv("minirocket_basic_motions_metrics.csv", index=False)


print("\nMetrics:")
print(stats_df)


print("\nAll predictions:")
for i in range(len(y_test)):
    print(f"Sample {i}: Predicted = {y_pred[i]}, Actual = {y_test[i]}")


best_alpha = pipeline.named_steps["ridgeclassifiercv"].alpha_
print(f"\nBest alpha selected by RidgeClassifierCV: {best_alpha}")

print("\nMisclassified samples:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f"Sample {i}: Predicted = {y_pred[i]}, Actual = {y_test[i]}")


#-----------------------------------------------------------------------------
# Load Basic Motions dataset
X_train, y_train = load_basic_motions(split="train", return_X_y=True)
X_test, y_test = load_basic_motions(split="test", return_X_y=True)

# ============================================
# PRINT STRUCTURE OF 1 SAMPLE AS MATRIX
# ============================================

sample_idx = 0  # change index if you want to view a different sample

print("="*70)
print(f"DATA STRUCTURE SAMPLE {sample_idx} - BASIC MOTIONS")
print("="*70)

# Get sample
sample = X_train.iloc[sample_idx]
label = y_train[sample_idx]

print(f"\nLabel: {label}")
print(f"Type: {type(sample)}")
print(f"Number of dimensions: {len(sample)}")
print(f"Dimension names: {sample.index.tolist()}")

# ============================================
# 1. PRINT AS MATRIX (100×6)
# ============================================

print("\n" + "="*70)
print(f"DATA MATRIX (100 rows × 6 columns)")
print("="*70)

# Convert sample to numpy array (100×6)
matrix = np.zeros((100, 6))
dim_names = []

for i, dim in enumerate(sample.index):
    series = sample[dim]
    matrix[:, i] = series.values
    dim_names.append(dim)

print(f"\nShape: {matrix.shape}")
print(f"Columns: {dim_names}")

# Print matrix
print("\nMatrix (100 rows × 6 columns):")
print("-"*70)

# Header
header = "Time:".ljust(8) + " ".join([f"{dim:^10}" for dim in dim_names])
print(header)
print("-"*70)

# Print first 10 rows
for t in range(10):
    row = f"{t:4d}".ljust(8)
    for col in range(6):
        row += f"{matrix[t, col]:+10.4f}"
    print(row)

print("...")

# Print last 5 rows
for t in range(95, 100):
    row = f"{t:4d}".ljust(8)
    for col in range(6):
        row += f"{matrix[t, col]:+10.4f}"
    print(row)

# Length of each sample
print("="*70)
print(f"LENGTH OF SAMPLES OF TRAINING SET - BASIC MOTIONS")
print("="*70)
max_length = max([X_train.iloc[i, 0].shape[0] for i in range(len(y_train))])
print(f"Max length: {max_length}")
for i in range(len(y_train)):
    length = X_train.iloc[i, 0].shape[0]
    print(f"Sample {i}: length = {length}")

# ============================================
# 2. PRINT AS PANDAS DATAFRAME
# ============================================

# print("\n" + "="*70)
# print(f"DATAFRAME (100 rows × 6 columns)")
# print("="*70)

# df_matrix = pd.DataFrame(matrix, columns=dim_names)
# df_matrix.index.name = "Time"
# print(df_matrix.head(10))
# print("\n...")
# print(df_matrix.tail(5))

# ============================================
# 3. MATRIX STATISTICS
# ============================================

# print("\n" + "="*70)
# print("STATISTICS BY COLUMN (DIMENSION)")
# print("="*70)

# print(f"\n{'Dimension':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
# print("-"*60)

# for i, dim in enumerate(dim_names):
#     col_data = matrix[:, i]
#     print(f"{dim:<12} {col_data.mean():12.4f} {col_data.std():12.4f} {col_data.min():12.4f} {col_data.max():12.4f}")

# ============================================
# 4. PRINT ALL 100 ROWS (OPTIONAL)
# ============================================
# print("\n" + "="*70)
# print("ALL 100 ROWS OF DATA")
# print("="*70)
# for t in range(100):
#     row = f"{t:4d}".ljust(8)
#     for col in range(6):
#         row += f"{matrix[t, col]:+10.4f}"
#     print(row)