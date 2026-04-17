# Implementation Plan

- [ ] 1. 调整论文模板骨架并接入实验章节
  - 修改 `SCUT-thesis/main.tex`，在现有 `chap04` 后接入 `\include{docs/chap05}`，保持封面、声明、摘要、目录、正文、结论、参考文献、致谢、附录的既有顺序不变。
  - 新建 `SCUT-thesis/docs/chap05.tex`，初始化“实验设计与结果分析”章节标题、节结构、标签与交叉引用占位。
  - 如模板当前章节标题或 `\include` 顺序与设计文档不一致，同步修正章节命名，使后续内容文件全部能挂接到模板入口上。

- [ ] 2. 填写前置部分：封面信息、摘要与关键词
  - [ ] 2.1 更新 `SCUT-thesis/docs/info.tex`
    - 写入论文题名《基于联邦学习的车联网入侵检测系统设计与实现》的中英文标题字段。
    - 将作者、学号、导师、电话、邮箱等未知信息改成明确的待补占位形式，避免保留模板示例中的虚假个人信息。
  - [ ] 2.2 更新 `SCUT-thesis/docs/abstract.tex`
    - 编写中文摘要、英文摘要和中英文关键词，内容需与项目代码、实验目标和最终结论保持一致。
    - 摘要中覆盖研究背景、系统方法、改进策略、实验结论与应用价值，不引入仓库中无法证实的结果。
  - [ ] 2.3 修正前置部分的 LaTeX 细节
    - 检查摘要命令、关键词分隔、特殊字符转义和前置部分引用格式，消除会导致模板编译失败的语法问题。

- [ ] 3. 编写绪论与相关技术章节
  - [ ] 3.1 更新 `SCUT-thesis/docs/chap01.tex`
    - 按“研究背景与意义 → 国内外研究现状 → 研究内容与目标 → 论文结构安排”的顺序撰写绪论。
    - 基于 `README.md`、实验说明和题目组织正文；对暂未从开题报告提取出的文献与任务书原文使用受控占位。
  - [ ] 3.2 更新 `SCUT-thesis/docs/chap02.tex`
    - 编写车联网、入侵检测、联邦学习、FedAvg/FedProx、CNN-LSTM、节点选择、Top-K 稀疏化、定点量化与评价指标等内容。
    - 结合 `models.py`、`federated_learning.py`、`metrics.py` 中的真实实现术语组织技术原理描述。
  - [ ] 3.3 补齐章节内引用占位
    - 在 `chap01.tex` 与 `chap02.tex` 中插入后续可替换的参考文献引用键，保证正文先可落文，再由 `main.bib` 统一补全文献。

- [ ] 4. 编写系统分析、总体设计与系统实现章节
  - [ ] 4.1 更新 `SCUT-thesis/docs/chap03.tex`
    - 基于 `main.py`、`config.py`、`schemas.py`、`data_processor.py`、`federated_learning.py`、`models.py` 撰写系统目标、使用场景、总体架构、模块划分、数据流与运行流程。
    - 将“预处理 / 训练 / 测试”三种 CLI 模式、配置加载流程和输出产物目录明确写入系统设计部分。
  - [ ] 4.2 更新 `SCUT-thesis/docs/chap04.tex`
    - 按“数据预处理实现 → 客户端切分与状态建模 → 联邦训练与聚合 → 参数压缩与通信统计 → 模型评估与报告落盘”的顺序撰写实现章节。
    - 结合 `data_processor.py`、`federated_learning.py`、`models.py`、`runtime_utils.py`、`app_logging.py` 的函数与类行为描述关键实现机制。
  - [ ] 4.3 在设计与实现章节补充图表/伪代码占位
    - 为系统架构图、训练流程图、模块关系图、关键算法流程和参数表预留 LaTeX 图表与标签位置，便于后续继续增强排版。

- [ ] 5. 编写实验设计、结果分析与附录内容
  - [ ] 5.1 更新 `SCUT-thesis/docs/chap05.tex`
    - 基于 `docs/experimental_design.md`、`docs/experimental_results_summary.md`、`outputs/reports/federated_training_report.md`、`outputs/baseline_fedavg/reports/federated_training_report.md` 编写实验目标、实验设置、对照组、指标与结果分析。
    - 将改进方案与基线方案的关键数据写成表格或段落分析，明确通信开销下降、训练耗时变化、检测性能保持情况与隐私代理指标含义。
    - 对当前演示数据样本量较小、正式论文仍需在真实数据集上重复实验的限制写出规范说明。
  - [ ] 5.2 更新 `SCUT-thesis/docs/appendix1.tex`
    - 整理预处理、训练、测试命令示例，列出关键输出文件、实验配置和证据报告路径。
    - 将不适合放在正文中的附加表格、参数说明或结果文件说明移动到附录。
  - [ ] 5.3 保持实验章节与正文交叉引用一致
    - 检查实验表格、图名、章节标签和附录引用，避免新增 `chap05.tex` 后出现编号或引用断裂。

- [ ] 6. 完成结论、致谢、参考文献与编译修正
  - [ ] 6.1 更新 `SCUT-thesis/docs/conclusion.tex`
    - 总结系统设计、联邦训练改进策略、实验结果和工程价值，并将真实数据扩展、正式文献补齐等内容写入展望。
  - [ ] 6.2 更新 `SCUT-thesis/docs/ack.tex` 与 `SCUT-thesis/main.bib`
    - 在 `ack.tex` 写入通用且可继续个性化修改的致谢内容。
    - 在 `main.bib` 中先录入可从现有材料确认的参考文献条目，对尚未从开题报告准确提取的文献使用可替换占位键，保证正文引用结构完整。
  - [ ] 6.3 修正整套模板中的 LaTeX 语法与引用问题
    - 统一检查新增文件、交叉引用、文献键、章节标签、中文标点与特殊字符处理，修复会阻断论文模板编译的错误。
    - 保证所有新增或修改的 `.tex` / `.bib` 文件已接入 `SCUT-thesis/main.tex`，不存在孤立内容文件。