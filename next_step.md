## Problem 1 — Case 4 has no valid test windows

**Root cause:** The **Label 4** segment in the test file (`4_4`) contains only ~40 rows. After resampling to 64 Hz, this is ~38 timesteps — shorter than one window (128 steps), so **Pure Window produces zero samples** for this class.

**Fix:** Switch from Pure Window to **Majority Vote** for window labeling. Each window is assigned the label that appears most frequently within it, so short fault segments can still contribute at least one window.

**Trade-off:** Majority Vote introduces **label noise** at segment boundaries. Windows that straddle a transition (e.g. normal → fault) will be labeled by whichever phase is longer inside the window, which may not reflect the true signal. This is acceptable for short segments where Pure Window produces nothing.

---

## Problem 2 — aeon rejects training due to near-zero variance in channel 33

**Root cause:** When **stride is set to 1**, far more windows are generated, including many where channel 33 is completely flat (std ≤ 1e-07). This is likely a boolean signal that stays locked at 0 or 1 during certain fault states. After normalization the problem gets worse — if a channel's max equals its min, the scaler sets the **entire channel to 0**, making std exactly 0.

aeon performs a **variance check** before training and raises a `ValueError` if any channel in any window falls below this **threshold**.

**Fix:** After normalization, compute the standard deviation of each channel across all training samples and timesteps. **Drop** any channel whose std falls below a safe threshold (e.g. 1e-06) from both `X_train` and `X_test`.


**Trade-off:** Removing a channel means **permanently discarding that sensor's data**. In practice, a channel with near-zero variance across the entire training set carries no discriminative information, so this has no measurable impact on classification performance.