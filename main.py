"""车联网入侵检测系统命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from app_logging import setup_logging
from config import (
    AppConfig,
    DatasetConfig,
    PathConfig,
    TrainingConfig,
    build_app_config,
    load_config_file,
)
from data_processor import DataProcessor
from federated_learning import FedAvgProxOptimizer
from models import LightweightCNNLSTM
from runtime_utils import collect_run_metadata, save_json, set_global_seed
from schemas import EvalResult, PreprocessResult


LOGGER = logging.getLogger(__name__)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于联邦学习的车联网入侵检测系统")
    parser.add_argument("--config", type=str, default=None, help="TOML/JSON 配置文件路径")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "preprocess"])
    parser.add_argument("--rounds", type=int, default=50, help="联邦学习全局通信轮数")
    parser.add_argument("--clients", type=int, default=10, help="参与训练的车载节点数量")
    parser.add_argument("--data-dir", type=str, default="./data", help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="处理结果与模型输出目录")
    parser.add_argument("--batch-size", type=int, default=32, help="训练批大小")
    parser.add_argument("--epochs", type=int, default=1, help="本地训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="cpu", help="训练或推理设备")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--dataset-name", type=str, default="VeReMi & Car-Hacking", help="数据集名称")
    parser.add_argument("--label-column", type=str, default="label", help="标签列名")
    parser.add_argument("--validation-split", type=float, default=0.2, help="验证集划分比例")
    parser.add_argument("--client-fraction", type=float, default=0.5, help="每轮参与联邦训练的客户端比例")
    parser.add_argument("--fedprox-mu", type=float, default=0.0, help="FedProx 正则项系数")
    parser.add_argument("--topk-ratio", type=float, default=0.1, help="参数上传 Top-K 稀疏化比例")
    parser.add_argument("--quant-bits", type=int, default=8, help="参数量化位宽，支持 8/16/32")
    parser.add_argument("--selection-weight-compute", type=float, default=0.4, help="节点选择中算力评分权重")
    parser.add_argument("--selection-weight-battery", type=float, default=0.3, help="节点选择中电量评分权重")
    parser.add_argument("--selection-weight-channel", type=float, default=0.3, help="节点选择中信道评分权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--deterministic", action="store_true", help="启用确定性运行模式")
    parser.add_argument("--run-name", type=str, default="default", help="本次运行名称")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--json-logs", action="store_true", help="启用 JSON 格式日志")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def load_parser_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    if not config_args.config:
        return {}

    config_mapping = load_config_file(config_args.config)
    app_config = build_app_config(config_mapping)
    return {
        "config": config_args.config,
        "data_dir": str(app_config.path.data_dir),
        "output_dir": str(app_config.path.output_dir),
        "checkpoint": str(app_config.path.checkpoint_path) if app_config.path.checkpoint_path else None,
        "dataset_name": app_config.dataset.dataset_name,
        "label_column": app_config.dataset.label_column,
        "validation_split": app_config.dataset.validation_split,
        "rounds": app_config.training.global_rounds,
        "clients": app_config.training.num_clients,
        "epochs": app_config.training.local_epochs,
        "batch_size": app_config.training.batch_size,
        "lr": app_config.training.learning_rate,
        "device": app_config.training.device,
        "client_fraction": app_config.training.client_fraction,
        "fedprox_mu": app_config.training.fedprox_mu,
        "topk_ratio": app_config.training.compression_topk_ratio,
        "quant_bits": app_config.training.quantization_bits,
        "selection_weight_compute": app_config.training.selection_weight_compute,
        "selection_weight_battery": app_config.training.selection_weight_battery,
        "selection_weight_channel": app_config.training.selection_weight_channel,
        "seed": app_config.runtime.seed,
        "deterministic": app_config.runtime.deterministic,
        "run_name": app_config.runtime.run_name,
        "log_level": app_config.logging.level,
        "json_logs": app_config.logging.json_logs,
    }


def _was_option_provided(argv: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def build_configs(
    args: argparse.Namespace,
    *,
    explicit_checkpoint: bool = False,
    explicit_output_dir: bool = False,
) -> AppConfig:
    checkpoint_path = args.checkpoint
    if explicit_output_dir and not explicit_checkpoint:
        checkpoint_path = None

    return AppConfig(
        dataset=build_app_config(
            {
                "dataset": {
                    "dataset_name": args.dataset_name,
                    "label_column": args.label_column,
                    "validation_split": args.validation_split,
                }
            }
        ).dataset,
        path=build_app_config(
            {
                "path": {
                    "data_dir": args.data_dir,
                    "output_dir": args.output_dir,
                    "checkpoint_path": checkpoint_path,
                }
            }
        ).path,
        training=build_app_config(
            {
                "training": {
                    "global_rounds": args.rounds,
                    "num_clients": args.clients,
                    "local_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "device": args.device,
                    "client_fraction": args.client_fraction,
                    "fedprox_mu": args.fedprox_mu,
                    "compression_topk_ratio": args.topk_ratio,
                    "quantization_bits": args.quant_bits,
                    "selection_weight_compute": args.selection_weight_compute,
                    "selection_weight_battery": args.selection_weight_battery,
                    "selection_weight_channel": args.selection_weight_channel,
                }
            }
        ).training,
        logging=build_app_config(
            {
                "logging": {
                    "level": args.log_level,
                    "json_logs": args.json_logs,
                }
            }
        ).logging,
        runtime=build_app_config(
            {
                "runtime": {
                    "seed": args.seed,
                    "deterministic": args.deterministic,
                    "run_name": args.run_name,
                }
            }
        ).runtime,
    )


def load_preprocess_result(app_config: AppConfig) -> PreprocessResult:
    path_config = app_config.path
    label_column = app_config.dataset.label_column
    required_paths = [
        path_config.preprocessed_train_path,
        path_config.preprocessed_val_path,
        path_config.label_mapping_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "缺少预处理产物，请先执行 preprocess 模式。缺失文件: " + ", ".join(missing_paths)
        )

    train_dataset = pd.read_csv(path_config.preprocessed_train_path)
    validation_dataset = pd.read_csv(path_config.preprocessed_val_path)
    with path_config.label_mapping_path.open("r", encoding="utf-8") as file:
        label_mapping = json.load(file)

    feature_columns = [column for column in train_dataset.columns if column != label_column]
    return PreprocessResult(
        train_dataset_path=path_config.preprocessed_train_path,
        validation_dataset_path=path_config.preprocessed_val_path,
        label_mapping_path=path_config.label_mapping_path,
        feature_columns=feature_columns,
        label_column=label_column,
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        num_classes=len(label_mapping),
        dataset_name=app_config.dataset.dataset_name,
    )


def run_preprocess(dataset_config: DatasetConfig, path_config: PathConfig) -> PreprocessResult:
    processor = DataProcessor(dataset_config=dataset_config, path_config=path_config)
    result = processor.process_local_data()
    LOGGER.info(
        f"预处理完成，训练集 {result.train_size} 条，验证集 {result.validation_size} 条，"
        f"特征维度 {len(result.feature_columns)}。"
    )
    return result


def run_train(
    training_config: TrainingConfig,
    path_config: PathConfig,
    preprocess_result: PreprocessResult,
) -> list[dict]:
    optimizer = FedAvgProxOptimizer(training_config=training_config, path_config=path_config)
    history = optimizer.train(preprocess_result=preprocess_result)
    LOGGER.info(
        "联邦训练完成，共执行 %s 轮，全局模型已保存到 %s",
        len(history),
        path_config.resolve_checkpoint_path(training_config.checkpoint_name),
    )
    return history


def run_test(
    training_config: TrainingConfig,
    path_config: PathConfig,
    preprocess_result: PreprocessResult,
) -> EvalResult:
    checkpoint_path = path_config.resolve_checkpoint_path(training_config.checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    validation_dataset = pd.read_csv(preprocess_result.validation_dataset_path)
    model = LightweightCNNLSTM(
        input_dim=len(preprocess_result.feature_columns),
        num_classes=preprocess_result.num_classes,
        training_config=training_config,
        path_config=path_config,
    )
    result = model.evaluate(
        inputs=validation_dataset[preprocess_result.feature_columns].values,
        labels=validation_dataset[preprocess_result.label_column].values,
        checkpoint_path=checkpoint_path,
        batch_size=training_config.batch_size,
    )
    evaluation_path = save_json(path_config.evaluation_path, result.to_dict())
    LOGGER.info("评估完成，结果已保存到 %s", evaluation_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(load_parser_defaults(effective_argv))
    try:
        args = parser.parse_args(effective_argv)
        app_config = build_configs(
            args,
            explicit_checkpoint=_was_option_provided(effective_argv, "--checkpoint"),
            explicit_output_dir=_was_option_provided(effective_argv, "--output-dir"),
        )
        app_config.path.ensure_runtime_dirs()
        setup_logging(app_config.path, app_config.logging)
        set_global_seed(app_config.runtime.seed, deterministic=app_config.runtime.deterministic)
        save_json(app_config.path.config_snapshot_path, app_config.to_dict())
        save_json(
            app_config.path.metadata_path,
            collect_run_metadata(app_config, args.mode, effective_argv),
        )

        LOGGER.info("=== 启动车联网入侵检测系统 (模式: %s, run=%s) ===", args.mode, app_config.runtime.run_name)

        if args.mode == "preprocess":
            LOGGER.info("执行分布式数据本地预处理...")
            run_preprocess(dataset_config=app_config.dataset, path_config=app_config.path)
        elif args.mode == "train":
            LOGGER.info(
                "启动联邦学习协同训练 (节点数: %s, 轮数: %s)...",
                app_config.training.num_clients,
                app_config.training.global_rounds,
            )
            preprocess_result = load_preprocess_result(app_config)
            run_train(
                training_config=app_config.training,
                path_config=app_config.path,
                preprocess_result=preprocess_result,
            )
        elif args.mode == "test":
            LOGGER.info("执行轻量化本地入侵检测模型测试...")
            preprocess_result = load_preprocess_result(app_config)
            run_test(
                training_config=app_config.training,
                path_config=app_config.path,
                preprocess_result=preprocess_result,
            )
        return 0
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"参数或运行流程错误: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logging.getLogger(__name__).exception("发生未预期错误")
        print(f"未预期错误: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
