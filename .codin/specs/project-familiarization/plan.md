# Implementation Plan

- [ ] 1. 补齐共享配置与基础数据结构
  - 在 `config.py` 中新增 `DatasetConfig`、`TrainingConfig`、`PathConfig` 等 `dataclass`，统一管理数据目录、输出目录、训练轮数、客户端数量、批大小、学习率和设备参数。
  - 在 `schemas.py` 中新增 `ClientState`、`PreprocessResult`、`EvalResult` 等结构，作为 `data_processor.py`、`federated_learning.py`、`models.py` 之间的共享接口。
  - 修改 `main.py`、`data_processor.py`、`federated_learning.py`、`models.py`，让现有类优先从共享配置与结构中读取参数，而不是使用散落的硬编码值。
  - [ ] 1.1 在 `tests/test_config.py` 中为默认配置、参数覆盖和路径规范化编写单元测试。

- [ ] 2. 在 `data_processor.py` 中实现最小可运行的本地预处理流程
  - 扩展 `DataProcessor`，增加数据文件发现、CSV/表格读取、缺失值处理、标签标准化与基础特征提取逻辑，优先适配当前项目语义中的 VeReMi / Car-Hacking 风格表格数据。
  - 为 `process_local_data()` 增加输入路径校验、必要列校验、异常抛出与预处理产物保存逻辑，输出统一的 `PreprocessResult`。
  - 新增预处理产物落盘逻辑，至少保存训练集、验证集和标签映射到 `output_dir` 下，供训练与评估阶段直接复用。
  - [ ] 2.1 在 `tests/test_data_processor.py` 中使用临时目录和样例数据，验证数据读取、特征输出、标签映射和异常分支。

- [ ] 3. 在 `models.py` 中实现轻量化 CNN-LSTM 模型与评估接口
  - 将 `LightweightCNNLSTM` 改造成真实的 PyTorch 模型定义，补全 `build_model()`、前向传播和基础推理逻辑。
  - 在 `models.py` 中补充模型保存/加载、批量预测、损失计算和核心指标计算函数，返回统一的 `EvalResult`，替换当前固定打印指标。
  - 如当前文件职责过重，可新增 `metrics.py` 承载准确率、召回率、误报率等指标计算，再由 `models.py` 调用。
  - [ ] 3.1 在 `tests/test_models.py` 中为模型前向输出维度、检查点加载和指标计算编写单元测试。

- [ ] 4. 在 `federated_learning.py` 中实现最小联邦训练闭环
  - 基于 `ClientState` 与预处理结果重构 `FedAvgProxOptimizer`，补充客户端数据装载、本地训练调用、权重聚合与全局模型下发流程。
  - 去除 `train()` 中第 3 轮强制退出的硬编码逻辑，改为严格遵循 `global_rounds`，并将每轮结果写入输出目录。
  - 在 `federated_learning.py` 中加入最小可用的 FedAvg/FedProx 聚合逻辑与客户端选择策略，至少支持基于可用客户端集合的稳定采样。
  - [ ] 4.1 在 `tests/test_federated_learning.py` 中为客户端选择、权重聚合和轮次控制编写单元测试。

- [ ] 5. 在 `main.py` 中接通完整命令行工作流
  - 扩展 `argparse` 参数，至少增加 `--data-dir`、`--output-dir`、`--batch-size`、`--epochs`、`--lr`、`--device`、`--checkpoint` 等选项，并映射到共享配置对象。
  - 重构 `main.py` 的 `preprocess`、`train`、`test` 三个分支，使其串联真实预处理、训练、模型加载与评估逻辑，而不是仅调用打印函数。
  - 在 `main.py` 中统一异常出口和返回码，确保数据缺失、模型缺失、参数非法时给出清晰错误。
  - [ ] 5.1 在 `tests/test_main.py` 中为参数解析、模式路由和关键失败分支编写轻量测试，保证入口代码与前述模块完成集成。
