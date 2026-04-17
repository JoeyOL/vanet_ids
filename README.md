# vanet_ids

基于联邦学习的车联网入侵检测系统，提供 **数据预处理**、**联邦训练**、**本地评估** 三条完整命令行链路。

## 项目特点

- 支持表格风格的 VeReMi / Car-Hacking 数据预处理
- 提供最小可运行的 FedAvg / FedProx 联邦训练闭环
- 内置轻量化 CNN-LSTM PyTorch 模型
- 支持 TOML/JSON 配置文件、日志落盘、运行元数据与结果归档
- 支持动态节点选择、Top-K 稀疏压缩、定点量化与联邦训练证据报告
- 提供基础单元测试与 GitHub Actions CI

## 环境要求

- Python 3.10+
- 建议使用虚拟环境

## 安装

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## 快速开始

### 1. 准备配置

默认配置文件位于 `configs/default.toml`。

### 2. 数据预处理

```bash
./.venv/bin/python main.py --mode preprocess --config configs/default.toml
```

### 3. 联邦训练

```bash
./.venv/bin/python main.py --mode train --config configs/default.toml
```

### 4. 模型评估

```bash
./.venv/bin/python main.py --mode test --config configs/default.toml
```

## 目录说明

- `main.py`：CLI 入口与流程编排
- `config.py`：统一配置定义与配置文件加载
- `data_processor.py`：本地预处理
- `models.py`：轻量 CNN-LSTM 模型与评估接口
- `federated_learning.py`：联邦训练闭环
- `metrics.py`：分类指标计算
- `schemas.py`：跨模块共享结构
- `configs/`：示例配置
- `tests/`：单元测试

## 运行产物

默认会在 `output_dir` 下生成：

- `train_processed.csv`
- `val_processed.csv`
- `label_mapping.json`
- `training_history.json`
- `global_model.pt`
- `run_metadata.json`
- `config_snapshot.json`
- `logs/app.log`
- `reports/evaluation.json`
- `reports/federated_training_report.json`
- `reports/federated_training_report.md`

## 常用参数

- `--config`：配置文件路径，支持 `.toml` / `.json`
- `--data-dir`：原始数据目录
- `--output-dir`：输出目录
- `--rounds`：联邦轮数
- `--clients`：客户端数量
- `--client-fraction`：每轮采样客户端比例
- `--topk-ratio`：上传参数的 Top-K 稀疏化比例
- `--quant-bits`：参数量化位宽，支持 `8/16/32`
- `--selection-weight-compute`：节点选择算力权重
- `--selection-weight-battery`：节点选择电量权重
- `--selection-weight-channel`：节点选择信道权重
- `--epochs`：本地训练轮数
- `--lr`：学习率
- `--device`：`cpu` / `cuda` / `auto`
- `--seed`：随机种子
- `--deterministic`：启用确定性模式
- `--log-level`：日志级别
- `--json-logs`：输出 JSON 格式日志

CLI 参数会覆盖配置文件中的同名字段。

## 测试

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## 开发建议

- 优先使用配置文件管理实验参数
- 保留 `outputs/` 中的元数据与日志，便于复现和排障
- 在接入真实数据字段后，继续增强特征工程与评估报表

## 论文后续完善事项

当前仓库中的本科论文工程位于 `SCUT-thesis/`，正文、实验分析与附录已经完成初稿，并可正常编译生成 PDF。后续如果继续打磨论文，建议按以下方向补充和完善：

- **补全论文基本信息**：将 `SCUT-thesis/docs/info.tex` 中的作者、学号、学院、专业、指导教师等字段替换为正式信息，并确认封面与声明页内容无误。
- **补充正式参考文献**：将 `SCUT-thesis/main.bib` 中的占位条目逐步替换为真实、可检索的中英文文献，统一检查正文引用与参考文献表的一致性。
- **完善图表与系统插图**：根据章节中的占位说明补充系统架构图、模块关系图、训练流程图、通信收益图及实验结果可视化图表，增强论文表达的直观性。
- **扩展实验矩阵**：在当前演示性实验之外，进一步补充真实 VeReMi / Car-Hacking 数据集实验，增加不同客户端参与率、不同 Top-K 比例、不同量化位宽、FedProx 或集中式基线等对照结果。
- **增强实验可信度**：对关键实验进行多随机种子重复运行，补充均值、标准差和误报率分析，避免仅依据小样本单次结果得出过强结论。
- **优化论文文字与格式**：继续润色摘要、结论、致谢等章节语言，统一术语、图表标题、表格格式、变量写法与章节交叉引用，确保符合本科论文排版规范。
- **检查最终提交材料**：在正式提交前，重新编译 `SCUT-thesis/main.tex`，确认 `main.pdf`、附录命令、实验结果路径、截图与证据文件保持一致。

如果后续继续围绕论文工作推进，建议优先完成“正式文献替换 + 图表补全 + 扩展实验”这三项，它们对论文完整性和说服力提升最明显。
