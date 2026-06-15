
## 代码执行顺序与可观测节点

### Step 1 — 文件扫描与分组 `dataset.py: split_files()`

做了什么：扫描 `data/raw/` 目录，按 `TEST_FILE_MARKERS` 将 64 个文件分成训练组和测试组。

可输出的结果：
```
训练文件列表（52个）：
  2_1_timeSeriesStation60_...CSV  → Label 2
  2_2_timeSeriesStation60_...CSV  → Label 2
  ...
测试文件列表（12个）：
  2_4_timeSeriesStation60_...CSV  → Label 2
  3_4_timeSeriesStation60_...CSV  → Label 3
  ...
```

**理解要点：** 划分的单位是文件，不是行，这是防止数据泄露的核心设计。同一文件的相邻行之间高度相关，如果按行随机划分，训练集和测试集会几乎"共享"同一段信号。

---

### Step 2 — 单文件加载与频率估算 `preprocessing.py: load_csv() + _estimate_source_hz()`

做了什么：读取一个 CSV，从 Second 和 Nanosecond 两列计算相邻行的时间间隔中位数，得到该文件的实际采样频率。

可输出的结果：
```
文件: 2_1_timeSeriesStation60_...CSV
  原始行数: 1953
  估算频率: 65.0 Hz
  时间跨度: 约 30.0 秒
```

**理解要点：** 不能直接信任文件头或文件名提供的频率，因为 PLC 的采集周期在实际运行中会有漂移。用中位数而不是均值，是为了过滤掉偶发的长间隔（如 PLC 忙碌时跳过一个周期）。

---

### Step 3 — Label 映射 `preprocessing.py: process_file()`

做了什么：把原始 Label 列中的 15、16、17 替换为 1，得到映射后的标签数组。

可输出的结果：
```
文件: 2_1_timeSeriesStation60_...CSV
  原始 Label 分布: {0: 29行, 2: 270行, 1: 256行, 15: 181行, 16: 124行, 17: 1180行}
  映射后 Label 分布: {0: 29行, 2: 270行, 1: 1741行}
```

**理解要点：** 这一步决定了分类任务的实际目标。15/16/17 在物理上都是"故障已发生但系统还未完全恢复"的中间状态，合并为 Label=1 后语义更清晰，也避免了这三类因样本极少而拖低整体 Macro F1。

---

### Step 4 — 重采样 `preprocessing.py: _resample_array()`

做了什么：对每个特征列独立做多相滤波重采样，将信号从原始频率（如 65 Hz）变换到 64 Hz。Label 列用最近邻插值同步到新时间轴。

可输出的结果：
```
文件: 2_1_timeSeriesStation60_...CSV
  原始频率: 65.0 Hz → 目标: 64 Hz
  重采样比: up=32, down=33 (Fraction 近似)
  重采样前行数: 1953 → 重采样后: 1922
```

**理解要点：** 为什么不直接按行数滑窗？同样是 128 行，在 56 Hz 的文件里代表 2.28 秒，在 79 Hz 的文件里代表 1.62 秒。DTW 距离是时间形状的度量，时间尺度不一致会让同一类故障在不同文件里"看起来"速度不同，严重影响分类。统一到 64 Hz 后，128 步永远等于 2 秒。

---

### Step 5 — 零填充（仅短文件触发）`preprocessing.py: process_file()`

做了什么：对重采样后不足 128 步的文件，在右侧补零到 128 步。Label 填充为该文件的主要标签。

可输出的结果：
```
文件: 1_1_timeSeriesStation60_...CSV  [触发零填充]
  重采样后步数: 30 步
  零填充至: 128 步（补 98 步）
  主要 Label: 0
```

**理解要点：** 只有 `1_1`（31 行）会触发这个逻辑。零填充的物理含义是"信号突然消失"，对正常运行类（Label=0）来说影响尚可接受，因为静止状态下很多信号本来就接近零。

---

### Step 6 — 滑动窗口与 Pure Window 过滤 `preprocessing.py: _extract_windows()`

做了什么：以 128 步为窗口、64 步为步长滑过整个时间序列，对每个窗口检查标签是否单一，只保留标签纯净的窗口。

可输出的结果：
```
文件: 2_1_timeSeriesStation60_...CSV
  总候选窗口数: 28
  Pure Window 通过: 19
  被丢弃（混合标签）: 9
  各标签窗口数: {0: 0, 2: 4, 1: 15}
```

**理解要点：** 被丢弃的 9 个窗口正好落在 Label 切换的边界处（如从 0 变到 2 的那段）。这些窗口标签模糊，如果强行分配标签，会给模型提供矛盾的训练信号。代价是损失约 30% 的窗口，但换来的是干净的训练数据。

---

### Step 7 — 汇总所有文件，拼接训练/测试集 `dataset.py: build_dataset()`

做了什么：对所有训练文件依次执行 Step 2~6，将输出的窗口纵向拼接为大矩阵；测试文件同理。

可输出的结果：
```
X_train shape: (N_train, 44, 128)   → N_train 个窗口，每窗口 44 通道、128 时间步
y_train shape: (N_train,)
X_test  shape: (N_test,  44, 128)
y_test  shape: (N_test,)

训练集各 Label 窗口数:
  Label 0: xxx
  Label 1: xxx
  Label 2: xxx
  ...
```

**理解要点：** 拼接后的 X_train 就是 aeon 分类器的直接输入格式：`(n_samples, n_channels, n_timepoints)`。至此原始 CSV 已经完全转化为可以喂给模型的 numpy 数组。

---

### Step 8 — 归一化 `dataset.py: ChannelMinMaxScaler`

做了什么：对 X_train 的 44 个通道分别计算 min 和 max，将所有值缩放到 [0, 1]。同样的统计量应用到 X_test。

可输出的结果：
```
归一化前 X_train:
  通道 0 (Shuttle_in_Station): min=0.0, max=1.0
  通道 34 (M1 Ventil Ist-Druck): min=0.0, max=6.2
  通道 35 (M1 Ventil Soll-Druck): min=0.0, max=5.8
  ...
归一化后: 所有通道范围 [0.0, 1.0] ✓
```

**理解要点：** 压力传感器的量纲（bar 级别）和布尔信号（0/1）差了几个数量级。对 DTW 距离来说，量纲大的通道会主导距离计算，完全掩盖布尔信号的变化。归一化让每个通道在距离计算中权重相当。

---

### Step 9 — 模型训练 `train_eval.py: train_and_evaluate()`

做了什么：先跑 MiniROCKET，再跑 DTW-1NN，分别记录训练时间。

可输出的结果：
```
[MiniROCKET] 开始训练，训练集 N_train 个样本 ...
[MiniROCKET] 训练完成，耗时 17.89 s
  注意：首次运行含 Numba JIT 编译时间（约 10~20 s），不计入正式计时

[DTW-1NN]    开始训练（lazy，无实际计算）...
[DTW-1NN]    "训练"完成，耗时 0.06 s
```

**理解要点：** DTW-1NN 的 0.06 秒并不代表它"更快"——它只是把所有计算推迟到了推理阶段。代价在下一步看到。

---

### Step 10 — 推理与评估 `train_eval.py: train_and_evaluate()`

做了什么：在测试集上预测，计算 accuracy、Macro F1、Weighted F1、per-class F1、混淆矩阵，输出图表。

可输出的结果：
```
[MiniROCKET] 推理 N_test 个样本，耗时 0.199 s（0.72 ms/样本）
[DTW-1NN]    推理 N_test 个样本，耗时 13.255 s（48.2 ms/样本）

指标对比:
             MiniROCKET    DTW-1NN
Accuracy       98.55%       98.18%
Macro F1        0.993        0.906
Weighted F1     0.986        0.985

异常项: Label 4 (Pressure Mat) — 两个模型 F1 均为 0.000
```

---

## 分段交付的建议切割点

根据以上步骤，建议这样划分成三次交付：

**第一次交付（现在可做）**

Steps 1~3：文件扫描分组 + 频率估算 + Label 分布统计

输出：文件清单、每个文件的频率、Label 切换统计表。这部分完全不依赖模型，是对数据集的客观描述，可以独立汇报。

**第二次交付**

Steps 4~8：重采样 + 滑窗 + 数据集构建 + 归一化

输出：每个文件切出的窗口数统计、最终 X_train / X_test 的 shape、各 Label 的样本分布表。这部分说明"数据是怎么准备好的"。

**第三次交付**

Steps 9~10：模型训练 + 评估

输出：训练时间、推理时间、指标对比表、混淆矩阵图。这是最终结果。

