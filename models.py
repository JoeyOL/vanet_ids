"""轻量化入侵检测模型定义。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from config import PathConfig, TrainingConfig
from metrics import calculate_classification_metrics
from schemas import EvalResult


LOGGER = logging.getLogger(__name__)


class LightweightCNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 2,
        conv_channels: int = 16,
        lstm_hidden_size: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        training_config: TrainingConfig | None = None,
        path_config: PathConfig | None = None,
    ):
        super().__init__()
        self.training_config = training_config or TrainingConfig()
        self.path_config = path_config or PathConfig()
        self.model_name = "Lightweight CNN-LSTM"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.device_obj = self._resolve_device(self.training_config.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.build_model()
        self.to(self.device_obj)

    def build_model(self) -> None:
        LOGGER.info("构建 %s 模型结构 (1D-CNN 特征提取 + LSTM 时序分析)...", self.model_name)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.conv_channels),
            nn.Conv1d(
                in_channels=self.conv_channels,
                out_channels=self.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.temporal_encoder = nn.LSTM(
            input_size=self.conv_channels,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.lstm_hidden_size, self.num_classes),
        )

    def forward(self, inputs: Tensor | np.ndarray | Sequence[Sequence[float]]) -> Tensor:
        tensor_inputs = self._prepare_inputs(inputs)
        features = self.feature_extractor(tensor_inputs)
        temporal_features = features.transpose(1, 2)
        lstm_output, _ = self.temporal_encoder(temporal_features)
        return self.classifier(lstm_output[:, -1, :])

    def compute_loss(self, logits: Tensor, targets: Tensor | np.ndarray | Sequence[int]) -> Tensor:
        if isinstance(targets, torch.Tensor):
            tensor_targets = targets.to(device=logits.device, dtype=torch.long)
        else:
            tensor_targets = torch.tensor(np.asarray(targets).copy(), dtype=torch.long, device=logits.device)
        return self.loss_fn(logits, tensor_targets)

    def predict_batch(
        self,
        inputs: Tensor | np.ndarray | Sequence[Sequence[float]],
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        effective_batch_size = batch_size or self.training_config.batch_size
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32)
        dataset = TensorDataset(tensor_inputs)
        predictions: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []

        self.eval()
        with torch.no_grad():
            for batch_inputs, in DataLoader(dataset, batch_size=effective_batch_size):
                logits = self.forward(batch_inputs)
                batch_probabilities = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(batch_probabilities, dim=1)
                predictions.append(batch_predictions.cpu().numpy())
                probabilities.append(batch_probabilities.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(probabilities)

    def save_checkpoint(
        self,
        checkpoint_path: str | Path | None = None,
        extra_state: dict | None = None,
    ) -> Path:
        self.path_config.ensure_output_dir()
        resolved_path = Path(
            checkpoint_path or self.path_config.resolve_checkpoint_path(self.training_config.checkpoint_name)
        ).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "conv_channels": self.conv_channels,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "extra_state": extra_state or {},
        }
        torch.save(checkpoint_payload, resolved_path)
        return resolved_path

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> dict:
        resolved_path = Path(
            checkpoint_path or self.path_config.resolve_checkpoint_path(self.training_config.checkpoint_name)
        ).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"模型检查点不存在: {resolved_path}")

        checkpoint = torch.load(resolved_path, map_location=self.device_obj)
        if checkpoint.get("input_dim") != self.input_dim or checkpoint.get("num_classes") != self.num_classes:
            self.input_dim = int(checkpoint.get("input_dim", self.input_dim))
            self.num_classes = int(checkpoint.get("num_classes", self.num_classes))
            self.conv_channels = int(checkpoint.get("conv_channels", self.conv_channels))
            self.lstm_hidden_size = int(checkpoint.get("lstm_hidden_size", self.lstm_hidden_size))
            self.lstm_layers = int(checkpoint.get("lstm_layers", self.lstm_layers))
            self.dropout = float(checkpoint.get("dropout", self.dropout))
            self.build_model()
            self.to(self.device_obj)

        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        return checkpoint

    def evaluate(
        self,
        inputs: Tensor | np.ndarray | Sequence[Sequence[float]] | None = None,
        labels: Tensor | np.ndarray | Sequence[int] | None = None,
        checkpoint_path: str | Path | None = None,
        batch_size: int | None = None,
    ) -> EvalResult:
        LOGGER.info("加载全局模型参数...")
        LOGGER.info("使用设备: %s", self.device_obj)

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        if inputs is None or labels is None:
            LOGGER.info("在车载节点执行本地实时入侵检测...")
            return EvalResult(details={"message": "未提供评估数据"})

        LOGGER.info("在车载节点执行本地实时入侵检测...")
        start_time = time.perf_counter()
        predictions, _ = self.predict_batch(inputs, batch_size=batch_size)
        logits = self.forward(inputs)
        loss = self.compute_loss(logits, labels).item()
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = calculate_classification_metrics(labels, predictions, loss=loss, latency_ms=latency_ms)
        LOGGER.info(
            "评估指标: 检测率=%.1f%%, 误报率=%.1f%%, 推理延迟=%.2fms",
            result.recall * 100,
            result.false_positive_rate * 100,
            result.latency_ms,
        )
        return result

    def _prepare_inputs(self, inputs: Tensor | np.ndarray | Sequence[Sequence[float]]) -> Tensor:
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device_obj)
        if tensor_inputs.ndim == 1:
            tensor_inputs = tensor_inputs.unsqueeze(0)

        if tensor_inputs.ndim == 2:
            if tensor_inputs.shape[1] != self.input_dim:
                raise ValueError(
                    f"输入特征维度不匹配，期望 {self.input_dim}，实际 {tensor_inputs.shape[1]}"
                )
            tensor_inputs = tensor_inputs.unsqueeze(1)
        elif tensor_inputs.ndim == 3:
            if tensor_inputs.shape[1] == 1 and tensor_inputs.shape[2] == self.input_dim:
                pass
            else:
                raise ValueError("仅支持形状为 [batch, features] 或 [batch, 1, features] 的输入")
        else:
            raise ValueError("模型输入必须为二维或三维张量")

        return tensor_inputs

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested_device)
