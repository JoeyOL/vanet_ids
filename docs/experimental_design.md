## 车联网联邦入侵检测实验方案

### 1. 实验目标

- 验证轻量化 `CNN-LSTM` 在车联网入侵检测场景中的可用性。
- 验证改进联邦训练策略是否能同时降低通信开销，并保持可接受的检测性能。
- 形成可直接用于毕业论文“实验设计”和“结果分析”章节的标准化流程。

### 2. 核心研究问题与假设

- **H1**：采用三维度节点选择（算力 / 电量 / 信道）后，平均轮次训练时延低于随机或无差别参与方案。
- **H2**：采用 `Top-K` 稀疏化与定点量化后，总通信量显著低于未压缩联邦训练方案。
- **H3**：在存在压缩与节点筛选的情况下，模型 `Accuracy`、`Recall`、`F1` 不出现不可接受下降。
- **H4**：由于原始数据始终保留在本地，系统满足“数据不出车端”的隐私约束；结合压缩率可得到更高的隐私代理分数。

### 3. 实验对象与数据来源

- **目标数据集**：VeReMi、Car-Hacking 或其课程/论文可获得的等价 CAN/V2X 日志。
- **本项目内置复现实验数据**：`data/demo_vehicular.csv`
- **标签形式**：建议至少区分 `benign` 与 `attack`；若数据允许，可扩展为 `DoS`、`Spoofing`、`Replay` 等多分类。

### 4. 对照组与变量设置

### 4.1 对照组

- **Baseline-A：集中式训练**
  - 单机直接训练同一 `CNN-LSTM`，作为“理想上界”参考。
- **Baseline-B：标准 FedAvg**
  - 全量可用节点参与；不做节点评分；不做参数压缩。
- **Baseline-C：FedProx**
  - 全量节点参与；设置 `fedprox_mu > 0`；不做参数压缩。

### 4.2 改进方案组

- **Proposed-1：动态节点选择**
  - 使用算力 / 电量 / 信道三维评分；加入公平性惩罚。
- **Proposed-2：动态节点选择 + Top-K 压缩**
  - 固定 `quantization_bits=32`，仅验证稀疏化贡献。
- **Proposed-3：动态节点选择 + Top-K + 量化**
  - 完整方案，建议 `topk_ratio=0.1~0.2`，`quantization_bits=8/16`。

### 4.3 自变量

- **联邦轮数**：`5 / 10 / 20 / 50`
- **客户端数量**：`4 / 8 / 10`
- **客户端参与率**：`0.5 / 0.75 / 1.0`
- **FedProx 系数**：`0 / 0.001 / 0.01`
- **Top-K 比例**：`0.05 / 0.1 / 0.2 / 0.5 / 1.0`
- **量化位宽**：`8 / 16 / 32`
- **节点选择权重**：
  - 计算优先：`0.5 / 0.3 / 0.2`
  - 均衡模式：`0.4 / 0.3 / 0.3`
  - 通道优先：`0.3 / 0.2 / 0.5`

### 4.4 因变量

- **检测性能**：`Accuracy`、`Precision`、`Recall`、`F1-score`、`False Positive Rate`
- **通信性能**：
  - 原始通信量 `total_original_communication_bytes`
  - 压缩后通信量 `total_compressed_communication_bytes`
  - 总体通信压缩率 `overall_communication_reduction_ratio`
- **计算性能**：
  - 总训练时间 `total_training_time_ms`
  - 平均轮次耗时 `mean_round_duration_ms`
  - 平均本地训练耗时 `mean_local_training_time_ms`
- **隐私代理指标**：`mean_privacy_proxy_score`

### 5. 实验执行流程

### 5.1 数据预处理

```bash
./.venv/bin/python main.py --mode preprocess --config configs/default.toml
```

产物：

- `outputs/train_processed.csv`
- `outputs/val_processed.csv`
- `outputs/label_mapping.json`

### 5.2 联邦训练实验

#### 标准改进方案示例

```bash
./.venv/bin/python main.py --mode train --config configs/default.toml --rounds 3 --clients 4 --epochs 2 --batch-size 8 --client-fraction 0.75 --topk-ratio 0.2 --quant-bits 8 --selection-weight-compute 0.4 --selection-weight-battery 0.3 --selection-weight-channel 0.3 --run-name evidence_demo
```

#### Baseline FedAvg 示例

```bash
./.venv/bin/python main.py --mode train --config configs/default.toml --rounds 3 --clients 4 --epochs 2 --batch-size 8 --client-fraction 1.0 --topk-ratio 1.0 --quant-bits 32 --selection-weight-compute 1.0 --selection-weight-battery 0.0 --selection-weight-channel 0.0 --run-name baseline_fedavg
```

### 5.3 模型评估

```bash
./.venv/bin/python main.py --mode test --config configs/default.toml --rounds 3 --clients 4 --epochs 2 --batch-size 8 --run-name evidence_demo
```

### 6. 结果采集与文件映射

- **训练过程明细**：`outputs/training_history.json`
- **联邦训练证据报告（JSON）**：`outputs/reports/federated_training_report.json`
- **联邦训练证据报告（Markdown）**：`outputs/reports/federated_training_report.md`
- **测试评估结果**：`outputs/reports/evaluation.json`
- **运行元数据**：`outputs/run_metadata.json`
- **配置快照**：`outputs/config_snapshot.json`

### 7. 论文建议图表

- **图 1**：不同方案下 `Accuracy / Recall / F1` 对比柱状图
- **图 2**：不同 `Top-K` 比例下的通信压缩率折线图
- **图 3**：平均轮次耗时与客户端参与率关系图
- **图 4**：隐私代理分数与压缩率关系图
- **表 1**：各方案总体性能汇总表
- **表 2**：不同权重配置下节点选择结果统计表

### 8. 统计分析建议

- 每组实验至少运行 **3 次**，报告均值与标准差。
- 若论文要求显著性，可对 `Accuracy`、`Recall`、`通信压缩率` 做 `t-test` 或非参数检验。
- 当性能下降不明显但通信节省明显时，应突出“综合收益”而非单点精度。

### 9. 毕业论文撰写建议

- **系统实现章节**：重点写“节点选择策略、参数压缩策略、证据报告生成机制”。
- **实验章节**：按“实验环境 → 参数设置 → 对照组 → 指标 → 结果分析”展开。
- **结论章节**：突出“在保证检测性能的同时降低通信开销，并满足数据不出本地的隐私要求”。
