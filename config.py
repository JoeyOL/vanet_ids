from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _normalize_optional_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    return _normalize_path(value)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    resolved_path = _normalize_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {resolved_path}")

    if resolved_path.suffix.lower() == ".toml":
        with resolved_path.open("rb") as file:
            payload = tomllib.load(file)
            return _resolve_relative_paths(payload, resolved_path.parent)
    if resolved_path.suffix.lower() == ".json":
        with resolved_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
            return _resolve_relative_paths(payload, resolved_path.parent)
    raise ValueError(f"不支持的配置文件格式: {resolved_path.suffix}")


def _resolve_relative_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    path_section = dict(payload.get("path", {}))
    for key in ("data_dir", "output_dir", "checkpoint_path"):
        value = path_section.get(key)
        if value in (None, ""):
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            path_section[key] = str((base_dir / candidate).resolve())

    resolved_payload = dict(payload)
    resolved_payload["path"] = path_section
    return resolved_payload


@dataclass(slots=True)
class PathConfig:
    data_dir: Path = field(default_factory=lambda: _normalize_path("./data"))
    output_dir: Path = field(default_factory=lambda: _normalize_path("./outputs"))
    checkpoint_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.data_dir = _normalize_path(self.data_dir)
        self.output_dir = _normalize_path(self.output_dir)
        self.checkpoint_path = _normalize_optional_path(self.checkpoint_path)

    @property
    def preprocessed_train_path(self) -> Path:
        return self.output_dir / "train_processed.csv"

    @property
    def preprocessed_val_path(self) -> Path:
        return self.output_dir / "val_processed.csv"

    @property
    def label_mapping_path(self) -> Path:
        return self.output_dir / "label_mapping.json"

    @property
    def training_history_path(self) -> Path:
        return self.output_dir / "training_history.json"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def log_file_path(self) -> Path:
        return self.logs_dir / "app.log"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "run_metadata.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.output_dir / "config_snapshot.json"

    @property
    def evaluation_path(self) -> Path:
        return self.reports_dir / "evaluation.json"

    @property
    def federated_report_path(self) -> Path:
        return self.reports_dir / "federated_training_report.json"

    @property
    def federated_report_markdown_path(self) -> Path:
        return self.reports_dir / "federated_training_report.md"

    def resolve_checkpoint_path(self, fallback_name: str = "global_model.pt") -> Path:
        return self.checkpoint_path or (self.output_dir / fallback_name)

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def ensure_runtime_dirs(self) -> None:
        self.ensure_output_dir()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_parent = self.resolve_checkpoint_path().parent
        checkpoint_parent.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DatasetConfig:
    dataset_name: str = "VeReMi & Car-Hacking"
    label_column: str = "label"
    validation_split: float = 0.2
    random_seed: int = 42
    supported_extensions: tuple[str, ...] = (".csv", ".tsv", ".xlsx", ".xls")

    def __post_init__(self) -> None:
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split 必须位于 0 和 1 之间")


@dataclass(slots=True)
class TrainingConfig:
    global_rounds: int = 50
    num_clients: int = 10
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"
    checkpoint_name: str = "global_model.pt"
    client_fraction: float = 0.5
    fedprox_mu: float = 0.0
    compression_topk_ratio: float = 0.1
    quantization_bits: int = 8
    selection_weight_compute: float = 0.4
    selection_weight_battery: float = 0.3
    selection_weight_channel: float = 0.3

    def __post_init__(self) -> None:
        if self.global_rounds <= 0:
            raise ValueError("global_rounds 必须大于 0")
        if self.num_clients <= 0:
            raise ValueError("num_clients 必须大于 0")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs 必须大于 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction 必须位于 0 和 1 之间")
        if self.fedprox_mu < 0:
            raise ValueError("fedprox_mu 不能为负数")
        if not 0 < self.compression_topk_ratio <= 1:
            raise ValueError("compression_topk_ratio 必须位于 0 和 1 之间")
        if self.quantization_bits not in {8, 16, 32}:
            raise ValueError("quantization_bits 仅支持 8、16、32")
        selection_weight_sum = (
            self.selection_weight_compute + self.selection_weight_battery + self.selection_weight_channel
        )
        if selection_weight_sum <= 0:
            raise ValueError("节点选择权重之和必须大于 0")
        if min(
            self.selection_weight_compute,
            self.selection_weight_battery,
            self.selection_weight_channel,
        ) < 0:
            raise ValueError("节点选择权重不能为负数")


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    json_logs: bool = False
    log_to_file: bool = True

    def __post_init__(self) -> None:
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        normalized_level = str(self.level).upper()
        if normalized_level not in allowed_levels:
            raise ValueError(f"不支持的日志级别: {self.level}")
        self.level = normalized_level


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = False
    run_name: str = "default"


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    path: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": _serialize(asdict(self.dataset)),
            "path": _serialize(asdict(self.path)),
            "training": _serialize(asdict(self.training)),
            "logging": _serialize(asdict(self.logging)),
            "runtime": _serialize(asdict(self.runtime)),
        }


def build_app_config(config_mapping: dict[str, Any] | None = None) -> AppConfig:
    payload = config_mapping or {}
    return AppConfig(
        dataset=DatasetConfig(**payload.get("dataset", {})),
        path=PathConfig(**payload.get("path", {})),
        training=TrainingConfig(**payload.get("training", {})),
        logging=LoggingConfig(**payload.get("logging", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
    )
