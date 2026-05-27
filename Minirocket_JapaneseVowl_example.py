import numpy as np
import pandas as pd
from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_japanese_vowels
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable

# Load Japanese Vowels dataset
X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])
print("Number of dimensions:", X_train.shape[1])
print("Length of sample 0, dim 0:", X_train.iloc[0, 0].shape[0])
print("Length of sample 1, dim 0:", X_train.iloc[1, 0].shape[0])

seed = 42
np.random.seed(seed)

stats = []

start = timer()
pipeline = make_pipeline(
    MiniRocketMultivariateVariable(
        pad_value_short_series=-10.0,
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
    "JapaneseVowels",
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

stats_df.to_csv("minirocket_japanese_vowels_metrics.csv", index=False)

print("\nMetrics:")
print(stats_df)

print("\nAll predictions:")
for i in range(len(y_test)):
    print(f"Sample {i}: Predicted = Speaker #{y_pred[i]}, Actual = Speaker #{y_test[i]}")

best_alpha = pipeline.named_steps["ridgeclassifiercv"].alpha_
print(f"\nBest alpha selected by RidgeClassifierCV: {best_alpha}")

print("\nMisclassified samples:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print(f"Sample {i}: Predicted = Speaker #{y_pred[i]}, Actual = Speaker #{y_test[i]}")

#------------------------------------------------------------------------
# Load Japanese Vowels dataset
X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

# ============================================
# PRINT STRUCTURE OF 1 SAMPLE AS MATRIX
# ============================================

sample_idx = 0  # change index if you want to view a different sample

print("="*70)
print(f"DATA STRUCTURE SAMPLE {sample_idx} - JAPANESE VOWELS")
print("="*70)

# Get sample
sample = X_train.iloc[sample_idx]
label = y_train[sample_idx]

print(f"\nLabel (Speaker ID): {label}")
print(f"Type: {type(sample)}")
print(f"Number of dimensions: {len(sample)}")
print(f"Dimension names: {sample.index.tolist()}")

# Length of each sample
print("="*70)
print(f"LENGTH OF SAMPLES OF TRAINING SET - JAPANESE VOWELS")

#max length of samples in the dataset
max_length = max([X_train.iloc[i, 0].shape[0] for i in range(len(y_train))])
print(f"Max length: {max_length}")

print("="*70)
for i in range(len(y_train)):
    length = X_train.iloc[i, 0].shape[0]
    print(f"Sample {i}: length = {length}")

# ============================================
# 5. PRINT ALL VALUES FOR EACH DIMENSION (OPTIONAL)
# ============================================
# print all values for each dimension
# print("\n" + "="*70)
# print("ALL VALUES FOR EACH DIMENSION")
# print("="*70)

# for i, dim in enumerate(sample.index):
#     series = sample[dim]
#     values = series.values
    
#     print(f"\n{dim} (length={len(values)}):")
#     for t in range(len(values)):
#         print(f"  {t:3d}: {values[t]:+10.4f}")