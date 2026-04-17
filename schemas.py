from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _serialize_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, list):
            serialized[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            serialized[key] = value
    return serialized


@dataclass(slots=True)
class ClientState:
    client_id: int
    train_data_path: Optional[Path] = None
    val_data_path: Optional[Path] = None
    num_samples: int = 0
    is_available: bool = True
    compute_capacity: float = 1.0
    battery_level: float = 1.0
    channel_quality: float = 1.0
    privacy_locality_score: float = 1.0
    last_loss: Optional[float] = None
    last_training_time_ms: Optional[float] = None
    last_selection_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.train_data_path is not None:
            self.train_data_path = _normalize_path(self.train_data_path)
        if self.val_data_path is not None:
            self.val_data_path = _normalize_path(self.val_data_path)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_mapping(asdict(self))


@dataclass(slots=True)
class PreprocessResult:
    train_dataset_path: Path
    validation_dataset_path: Path
    label_mapping_path: Path
    feature_columns: list[str] = field(default_factory=list)
    label_column: str = "label"
    train_size: int = 0
    validation_size: int = 0
    num_classes: int = 0
    dataset_name: str = ""

    def __post_init__(self) -> None:
        self.train_dataset_path = _normalize_path(self.train_dataset_path)
        self.validation_dataset_path = _normalize_path(self.validation_dataset_path)
        self.label_mapping_path = _normalize_path(self.label_mapping_path)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_mapping(asdict(self))


@dataclass(slots=True)
class EvalResult:
    loss: float = 0.0
    accuracy: float = 0.0
    recall: float = 0.0
    false_positive_rate: float = 0.0
    latency_ms: float = 0.0
    sample_count: int = 0
    precision: float = 0.0
    f1_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_mapping(asdict(self))
