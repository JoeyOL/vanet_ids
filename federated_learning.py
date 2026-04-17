from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import PathConfig, TrainingConfig
from models import LightweightCNNLSTM
from schemas import ClientState, PreprocessResult


LOGGER = logging.getLogger(__name__)


class FedAvgProxOptimizer:
    def __init__(
        self,
        training_config: TrainingConfig | None = None,
        path_config: PathConfig | None = None,
        client_states: list[ClientState] | None = None,
    ):
        self.training_config = training_config or TrainingConfig()
        self.path_config = path_config or PathConfig()
        self.client_states = client_states or [
            ClientState(client_id=client_id) for client_id in range(self.training_config.num_clients)
        ]

    def select_clients(self, round_index: int = 0) -> list[ClientState]:
        LOGGER.info("执行三维度动态节点选择策略 (算力、电量、信道状态)...")
        scored_clients = self.score_available_clients(round_index=round_index)
        available_clients = [entry["client"] for entry in scored_clients]
        if not available_clients:
            return []

        sample_size = min(
            len(available_clients),
            max(1, math.ceil(len(available_clients) * self.training_config.client_fraction)),
        )
        return available_clients[:sample_size]

    def score_available_clients(self, round_index: int = 0) -> list[dict[str, Any]]:
        available_clients = [client for client in self.client_states if client.is_available]
        weight_sum = (
            self.training_config.selection_weight_compute
            + self.training_config.selection_weight_battery
            + self.training_config.selection_weight_channel
        )
        scored_clients: list[dict[str, Any]] = []
        for client in available_clients:
            fairness_penalty = min(float(client.metadata.get("times_selected", 0)) * 0.05, 0.2)
            composite_score = (
                self.training_config.selection_weight_compute * client.compute_capacity
                + self.training_config.selection_weight_battery * client.battery_level
                + self.training_config.selection_weight_channel * client.channel_quality
            ) / weight_sum
            final_score = max(0.0, composite_score - fairness_penalty)
            client.last_selection_score = final_score
            scored_clients.append(
                {
                    "client": client,
                    "client_id": client.client_id,
                    "round": round_index + 1,
                    "compute_capacity": client.compute_capacity,
                    "battery_level": client.battery_level,
                    "channel_quality": client.channel_quality,
                    "privacy_locality_score": client.privacy_locality_score,
                    "fairness_penalty": fairness_penalty,
                    "composite_score": composite_score,
                    "selection_score": final_score,
                    "selection_reason": (
                        "按算力/电量/信道加权评分排序，并对频繁入选节点施加轻微公平性惩罚"
                    ),
                }
            )

        return sorted(
            scored_clients,
            key=lambda entry: (-entry["selection_score"], entry["client_id"]),
        )

    def aggregate_weights(
        self,
        client_updates: list[dict[str, torch.Tensor]],
        selected_clients: list[ClientState],
    ) -> dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("缺少客户端更新，无法执行聚合")

        total_samples = sum(max(client.num_samples, 1) for client in selected_clients)
        aggregated: dict[str, torch.Tensor] = {}
        for parameter_name in client_updates[0].keys():
            parameter_template = client_updates[0][parameter_name]
            if not torch.is_floating_point(parameter_template):
                aggregated[parameter_name] = parameter_template.detach().clone()
                continue

            weighted_sum = torch.zeros_like(parameter_template, dtype=torch.float32)
            for update, client in zip(client_updates, selected_clients):
                client_weight = max(client.num_samples, 1) / total_samples
                weighted_sum += update[parameter_name].detach().float() * client_weight
            aggregated[parameter_name] = weighted_sum.to(parameter_template.dtype)

        return aggregated

    def train(self, preprocess_result: PreprocessResult | None = None) -> list[dict]:
        self.path_config.ensure_runtime_dirs()
        LOGGER.info("初始化全局模型...")
        train_dataset, validation_dataset, preprocess_result = self._load_preprocessed_artifacts(
            preprocess_result
        )
        self.client_states = self._prepare_client_states(train_dataset)
        feature_columns = preprocess_result.feature_columns or [
            column for column in train_dataset.columns if column != preprocess_result.label_column
        ]

        global_model = LightweightCNNLSTM(
            input_dim=len(feature_columns),
            num_classes=preprocess_result.num_classes,
            training_config=self.training_config,
            path_config=self.path_config,
        )
        global_state = self._clone_state_dict(global_model.state_dict())
        training_history: list[dict] = []
        total_training_start = time.perf_counter()
        model_stats = self._compute_model_stats(global_model)

        for round_index in range(self.training_config.global_rounds):
            round_start = time.perf_counter()
            LOGGER.info("--- 第 %s/%s 轮全局通信 ---", round_index + 1, self.training_config.global_rounds)
            score_entries = self.score_available_clients(round_index=round_index)
            selected_clients = self.select_clients(round_index)
            if not selected_clients:
                raise RuntimeError("当前没有可用客户端参与训练")

            LOGGER.info("选中 %s 个活跃车载节点进行本地训练...", len(selected_clients))
            selected_score_map = {
                entry["client_id"]: self._sanitize_score_entry(entry)
                for entry in score_entries
                if entry["client_id"] in {c.client_id for c in selected_clients}
            }
            client_updates: list[dict[str, torch.Tensor]] = []
            local_losses: list[float] = []
            local_training_times: list[float] = []
            compression_records: list[dict[str, Any]] = []
            for client_state in selected_clients:
                updated_state, local_loss, local_training_time_ms = self._train_single_client(
                    client_state=client_state,
                    global_state=global_state,
                    feature_columns=feature_columns,
                    label_column=preprocess_result.label_column,
                    num_classes=preprocess_result.num_classes,
                )
                reconstructed_state, compression_stats = self._compress_client_update(
                    updated_state=updated_state,
                    global_state=global_state,
                )
                client_state.last_loss = local_loss
                client_state.last_training_time_ms = local_training_time_ms
                client_state.metadata["times_selected"] = client_state.metadata.get("times_selected", 0) + 1
                client_updates.append(reconstructed_state)
                local_losses.append(local_loss)
                local_training_times.append(local_training_time_ms)
                compression_records.append(
                    {
                        "client_id": client_state.client_id,
                        "num_samples": client_state.num_samples,
                        "local_loss": local_loss,
                        "local_training_time_ms": local_training_time_ms,
                        "selection_evidence": selected_score_map.get(client_state.client_id, {}),
                        "compression": compression_stats,
                    }
                )

            LOGGER.info("应用模型参数量化压缩与稀疏化...")
            LOGGER.info("全局模型参数安全聚合 (改进 FedAvg/FedProx)...")
            global_state = self.aggregate_weights(client_updates, selected_clients)
            global_model.load_state_dict(global_state)

            eval_result = self._evaluate_global_model(
                model=global_model,
                validation_dataset=validation_dataset,
                feature_columns=feature_columns,
                label_column=preprocess_result.label_column,
            )
            round_record = {
                "round": round_index + 1,
                "selected_clients": [client.client_id for client in selected_clients],
                "mean_local_loss": float(np.mean(local_losses)) if local_losses else 0.0,
                "mean_local_training_time_ms": float(np.mean(local_training_times)) if local_training_times else 0.0,
                "selection_evidence": [
                    self._sanitize_score_entry(entry) for entry in score_entries
                ],
                "client_records": compression_records,
                "communication": self._summarize_round_communication(
                    compression_records=compression_records,
                    available_client_count=len([client for client in self.client_states if client.is_available]),
                    model_stats=model_stats,
                ),
                "privacy_proxy_score": self._calculate_privacy_proxy_score(compression_records),
                "round_duration_ms": (time.perf_counter() - round_start) * 1000,
                "eval_result": eval_result.to_dict(),
            }
            training_history.append(round_record)
            self._persist_training_history(training_history)
            global_model.save_checkpoint(
                extra_state={
                    "round": round_index + 1,
                    "history": training_history,
                }
            )

        LOGGER.info("联邦学习协同训练完成，模型下发至各车载节点。")
        self._persist_federated_report(
            report=self._build_federated_report(
                training_history=training_history,
                preprocess_result=preprocess_result,
                model_stats=model_stats,
                total_training_time_ms=(time.perf_counter() - total_training_start) * 1000,
            )
        )
        return training_history

    def _load_preprocessed_artifacts(
        self,
        preprocess_result: PreprocessResult | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessResult]:
        effective_result = preprocess_result or PreprocessResult(
            train_dataset_path=self.path_config.preprocessed_train_path,
            validation_dataset_path=self.path_config.preprocessed_val_path,
            label_mapping_path=self.path_config.label_mapping_path,
        )

        required_paths = [
            effective_result.train_dataset_path,
            effective_result.validation_dataset_path,
            effective_result.label_mapping_path,
        ]
        missing_paths = [str(path) for path in required_paths if not Path(path).exists()]
        if missing_paths:
            raise FileNotFoundError(
                "缺少预处理产物，请先执行 preprocess 模式。缺失文件: " + ", ".join(missing_paths)
            )

        train_dataset = pd.read_csv(effective_result.train_dataset_path)
        validation_dataset = pd.read_csv(effective_result.validation_dataset_path)
        with effective_result.label_mapping_path.open("r", encoding="utf-8") as file:
            label_mapping = json.load(file)

        feature_columns = effective_result.feature_columns or [
            column for column in train_dataset.columns if column != effective_result.label_column
        ]
        effective_result.feature_columns = feature_columns
        effective_result.num_classes = effective_result.num_classes or len(label_mapping)
        effective_result.train_size = effective_result.train_size or len(train_dataset)
        effective_result.validation_size = effective_result.validation_size or len(validation_dataset)
        return train_dataset, validation_dataset, effective_result

    def _prepare_client_states(self, train_dataset: pd.DataFrame) -> list[ClientState]:
        if self.client_states and all(client.train_data_path for client in self.client_states):
            for client in self.client_states:
                if client.num_samples == 0 and client.train_data_path is not None:
                    client.num_samples = len(pd.read_csv(client.train_data_path))
                self._hydrate_client_resources(client)
            return self.client_states

        clients_dir = self.path_config.output_dir / "clients"
        clients_dir.mkdir(parents=True, exist_ok=True)
        split_indices = np.array_split(
            train_dataset.index,
            min(self.training_config.num_clients, len(train_dataset)),
        )
        prepared_clients: list[ClientState] = []

        for client_id, index_split in enumerate(split_indices):
            split_dataset = train_dataset.loc[index_split]
            if split_dataset.empty:
                continue

            client_train_path = clients_dir / f"client_{client_id}_train.csv"
            split_dataset.reset_index(drop=True).to_csv(client_train_path, index=False)
            prepared_clients.append(
                ClientState(
                    client_id=client_id,
                    train_data_path=client_train_path,
                    val_data_path=self.path_config.preprocessed_val_path,
                    num_samples=len(split_dataset),
                    compute_capacity=min(1.0, 0.55 + 0.08 * (client_id % 5)),
                    battery_level=min(1.0, 0.6 + 0.05 * ((client_id + 2) % 5)),
                    channel_quality=min(1.0, 0.58 + 0.06 * ((client_id + 1) % 5)),
                    privacy_locality_score=1.0,
                )
            )

        return prepared_clients

    def _train_single_client(
        self,
        client_state: ClientState,
        global_state: dict[str, torch.Tensor],
        feature_columns: list[str],
        label_column: str,
        num_classes: int,
    ) -> tuple[dict[str, torch.Tensor], float, float]:
        if client_state.train_data_path is None:
            raise ValueError(f"客户端 {client_state.client_id} 缺少训练数据路径")

        client_dataset = pd.read_csv(client_state.train_data_path)
        inputs = torch.tensor(client_dataset[feature_columns].values, dtype=torch.float32)
        targets = torch.tensor(client_dataset[label_column].values, dtype=torch.long)
        data_loader = DataLoader(
            TensorDataset(inputs, targets),
            batch_size=min(self.training_config.batch_size, len(client_dataset)),
            shuffle=True,
        )

        model = LightweightCNNLSTM(
            input_dim=len(feature_columns),
            num_classes=num_classes,
            training_config=self.training_config,
            path_config=self.path_config,
        )
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate)
        reference_parameters = {
            name: parameter.detach().clone().to(model.device_obj)
            for name, parameter in model.named_parameters()
        }

        train_start = time.perf_counter()
        epoch_losses: list[float] = []
        for _ in range(self.training_config.local_epochs):
            for batch_inputs, batch_targets in data_loader:
                optimizer.zero_grad()
                logits = model(batch_inputs)
                loss = model.compute_loss(logits, batch_targets)
                if self.training_config.fedprox_mu > 0:
                    loss = loss + self._compute_fedprox_penalty(model, reference_parameters)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

        updated_state = self._clone_state_dict(model.state_dict())
        client_state.num_samples = len(client_dataset)
        return (
            updated_state,
            float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            (time.perf_counter() - train_start) * 1000,
        )

    def _evaluate_global_model(
        self,
        model: LightweightCNNLSTM,
        validation_dataset: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
    ):
        validation_inputs = validation_dataset[feature_columns].values
        validation_targets = validation_dataset[label_column].values
        return model.evaluate(validation_inputs, validation_targets, batch_size=self.training_config.batch_size)

    def _compute_fedprox_penalty(
        self,
        model: LightweightCNNLSTM,
        reference_parameters: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=model.device_obj)
        for name, parameter in model.named_parameters():
            penalty = penalty + torch.sum((parameter - reference_parameters[name]) ** 2)
        return 0.5 * self.training_config.fedprox_mu * penalty

    def _persist_training_history(self, training_history: list[dict]) -> None:
        self.path_config.ensure_output_dir()
        with self.path_config.training_history_path.open("w", encoding="utf-8") as file:
            json.dump(training_history, file, ensure_ascii=False, indent=2)

    def _clone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in state_dict.items()
        }

    def _compress_client_update(
        self,
        updated_state: dict[str, torch.Tensor],
        global_state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        reconstructed_state: dict[str, torch.Tensor] = {}
        total_original_bytes = 0
        total_compressed_bytes = 0
        tensor_records: list[dict[str, Any]] = []

        for name, updated_tensor in updated_state.items():
            base_tensor = global_state[name]
            if not torch.is_floating_point(updated_tensor):
                original_bytes = updated_tensor.numel() * updated_tensor.element_size()
                total_original_bytes += original_bytes
                total_compressed_bytes += original_bytes
                reconstructed_state[name] = updated_tensor.detach().clone()
                tensor_records.append(
                    {
                        "name": name,
                        "original_bytes": original_bytes,
                        "compressed_bytes": original_bytes,
                        "retained_ratio": 1.0,
                    }
                )
                continue

            delta = (updated_tensor - base_tensor).detach().cpu().float()
            flat_delta = delta.flatten()
            numel = flat_delta.numel()
            topk = min(numel, max(1, math.ceil(numel * self.training_config.compression_topk_ratio)))
            original_bytes = numel * updated_tensor.element_size()

            if topk == numel:
                selected_indices = torch.arange(numel, dtype=torch.long)
            else:
                selected_indices = torch.topk(flat_delta.abs(), k=topk).indices
            selected_values = flat_delta[selected_indices]

            if topk == numel and self.training_config.quantization_bits == 32:
                total_original_bytes += original_bytes
                total_compressed_bytes += original_bytes
                reconstructed_state[name] = updated_tensor.detach().clone()
                tensor_records.append(
                    {
                        "name": name,
                        "original_bytes": original_bytes,
                        "compressed_bytes": original_bytes,
                        "retained_ratio": 1.0,
                    }
                )
                continue

            if self.training_config.quantization_bits == 32:
                dequantized_values = selected_values
                quantized_value_bytes = topk * 4
            else:
                quantized_dtype, quantized_value_bytes = self._resolve_quantization_dtype(topk)
                max_abs = float(selected_values.abs().max().item()) if topk else 0.0
                scale = max(max_abs / self._quantization_denominator(), 1e-8)
                quantized_values = torch.clamp(
                    torch.round(selected_values / scale),
                    -self._quantization_denominator(),
                    self._quantization_denominator(),
                ).to(quantized_dtype)
                dequantized_values = quantized_values.to(torch.float32) * scale
                quantized_value_bytes += 8

            compressed_bytes = topk * 4 + quantized_value_bytes + 16
            reconstructed_delta = torch.zeros_like(flat_delta, dtype=torch.float32)
            reconstructed_delta[selected_indices] = dequantized_values.to(torch.float32)
            reconstructed_state[name] = (base_tensor.detach().cpu().float() + reconstructed_delta.view_as(delta)).to(
                updated_tensor.dtype
            )

            total_original_bytes += original_bytes
            total_compressed_bytes += compressed_bytes
            tensor_records.append(
                {
                    "name": name,
                    "original_bytes": original_bytes,
                    "compressed_bytes": compressed_bytes,
                    "retained_ratio": float(topk / numel) if numel else 1.0,
                }
            )

        reduction_ratio = 1 - (total_compressed_bytes / total_original_bytes) if total_original_bytes else 0.0
        return reconstructed_state, {
            "original_bytes": total_original_bytes,
            "compressed_bytes": total_compressed_bytes,
            "reduction_ratio": reduction_ratio,
            "tensor_records": tensor_records,
            "compression_topk_ratio": self.training_config.compression_topk_ratio,
            "quantization_bits": self.training_config.quantization_bits,
        }

    def _resolve_quantization_dtype(self, topk: int) -> tuple[torch.dtype, int]:
        if self.training_config.quantization_bits == 8:
            return torch.int8, topk
        if self.training_config.quantization_bits == 16:
            return torch.int16, topk * 2
        return torch.float32, topk * 4

    def _quantization_denominator(self) -> int:
        return 2 ** (self.training_config.quantization_bits - 1) - 1

    def _hydrate_client_resources(self, client: ClientState) -> None:
        if client.compute_capacity <= 0:
            client.compute_capacity = 0.7
        if client.battery_level <= 0:
            client.battery_level = 0.7
        if client.channel_quality <= 0:
            client.channel_quality = 0.7
        if client.privacy_locality_score <= 0:
            client.privacy_locality_score = 1.0

    def _compute_model_stats(self, model: LightweightCNNLSTM) -> dict[str, Any]:
        total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        trainable_parameters = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        model_size_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "model_size_bytes": model_size_bytes,
        }

    def _sanitize_score_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in entry.items()
            if key != "client"
        }

    def _summarize_round_communication(
        self,
        compression_records: list[dict[str, Any]],
        available_client_count: int,
        model_stats: dict[str, Any],
    ) -> dict[str, Any]:
        original_bytes = int(sum(record["compression"]["original_bytes"] for record in compression_records))
        compressed_bytes = int(sum(record["compression"]["compressed_bytes"] for record in compression_records))
        reduction_ratio = 1 - (compressed_bytes / original_bytes) if original_bytes else 0.0
        baseline_full_participation_bytes = model_stats["model_size_bytes"] * max(available_client_count, 1)
        saving_against_full_baseline = (
            1 - (compressed_bytes / baseline_full_participation_bytes)
            if baseline_full_participation_bytes
            else 0.0
        )
        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "reduction_ratio": reduction_ratio,
            "baseline_full_participation_bytes": baseline_full_participation_bytes,
            "saving_against_full_participation": saving_against_full_baseline,
        }

    def _calculate_privacy_proxy_score(self, compression_records: list[dict[str, Any]]) -> float:
        if not compression_records:
            return 0.0
        scores: list[float] = []
        for record in compression_records:
            selection_evidence = record.get("selection_evidence", {})
            locality_score = float(selection_evidence.get("privacy_locality_score", 1.0))
            compression_gain = max(0.0, float(record["compression"].get("reduction_ratio", 0.0)))
            scores.append(min(1.0, 0.7 * locality_score + 0.3 * compression_gain))
        return float(np.mean(scores)) if scores else 0.0

    def _build_federated_report(
        self,
        training_history: list[dict],
        preprocess_result: PreprocessResult,
        model_stats: dict[str, Any],
        total_training_time_ms: float,
    ) -> dict[str, Any]:
        communication_original = sum(round_item["communication"]["original_bytes"] for round_item in training_history)
        communication_compressed = sum(round_item["communication"]["compressed_bytes"] for round_item in training_history)
        mean_round_duration_ms = float(np.mean([item["round_duration_ms"] for item in training_history])) if training_history else 0.0
        mean_privacy_proxy = float(np.mean([item["privacy_proxy_score"] for item in training_history])) if training_history else 0.0
        mean_training_time = float(
            np.mean([item["mean_local_training_time_ms"] for item in training_history])
        ) if training_history else 0.0
        final_eval = training_history[-1]["eval_result"] if training_history else {}
        return {
            "dataset": preprocess_result.dataset_name,
            "label_column": preprocess_result.label_column,
            "num_rounds": len(training_history),
            "num_clients": len(self.client_states),
            "selection_strategy": {
                "dimensions": ["compute_capacity", "battery_level", "channel_quality"],
                "weights": {
                    "compute_capacity": self.training_config.selection_weight_compute,
                    "battery_level": self.training_config.selection_weight_battery,
                    "channel_quality": self.training_config.selection_weight_channel,
                },
                "fairness_penalty": "每次入选增加 0.05，最高扣减 0.2",
            },
            "compression_strategy": {
                "method": "Top-K 稀疏化 + 定点量化",
                "topk_ratio": self.training_config.compression_topk_ratio,
                "quantization_bits": self.training_config.quantization_bits,
            },
            "model_stats": model_stats,
            "performance_summary": {
                "total_training_time_ms": total_training_time_ms,
                "mean_round_duration_ms": mean_round_duration_ms,
                "mean_local_training_time_ms": mean_training_time,
                "total_original_communication_bytes": communication_original,
                "total_compressed_communication_bytes": communication_compressed,
                "overall_communication_reduction_ratio": (
                    1 - (communication_compressed / communication_original)
                    if communication_original
                    else 0.0
                ),
                "mean_privacy_proxy_score": mean_privacy_proxy,
                "final_eval_result": final_eval,
            },
            "rounds": training_history,
            "evidence_notes": {
                "privacy_proxy_score": "代理指标，依据原始数据不出本地与上传更新压缩比例估计，不等同于差分隐私证明。",
                "communication_reduction": "与未压缩模型更新及全量节点参与基线比较。",
                "computation_overhead": "使用本地训练耗时与轮次总耗时衡量。",
            },
        }

    def _persist_federated_report(self, report: dict[str, Any]) -> None:
        with self.path_config.federated_report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        self.path_config.federated_report_markdown_path.write_text(
            self._render_federated_report_markdown(report),
            encoding="utf-8",
        )

    def _render_federated_report_markdown(self, report: dict[str, Any]) -> str:
        summary = report["performance_summary"]
        lines = [
            "# 联邦训练证据与性能报告",
            "",
            f"- 数据集：{report['dataset']}",
            f"- 轮数：{report['num_rounds']}",
            f"- 客户端数量：{report['num_clients']}",
            "",
            "## 改进策略证据",
            "",
            f"- 节点选择维度：{', '.join(report['selection_strategy']['dimensions'])}",
            f"- 节点选择权重：{report['selection_strategy']['weights']}",
            f"- 压缩方法：{report['compression_strategy']['method']}",
            f"- Top-K 比例：{report['compression_strategy']['topk_ratio']}",
            f"- 量化位宽：{report['compression_strategy']['quantization_bits']}",
            "",
            "## 性能摘要",
            "",
            f"- 总训练时间（ms）：{summary['total_training_time_ms']:.2f}",
            f"- 平均轮次耗时（ms）：{summary['mean_round_duration_ms']:.2f}",
            f"- 平均本地训练耗时（ms）：{summary['mean_local_training_time_ms']:.2f}",
            f"- 原始通信量（bytes）：{summary['total_original_communication_bytes']}",
            f"- 压缩后通信量（bytes）：{summary['total_compressed_communication_bytes']}",
            f"- 总体通信压缩率：{summary['overall_communication_reduction_ratio']:.4f}",
            f"- 平均隐私代理分数：{summary['mean_privacy_proxy_score']:.4f}",
            f"- 最终评估结果：{summary['final_eval_result']}",
            "",
            "## 说明",
            "",
            f"- 隐私代理指标说明：{report['evidence_notes']['privacy_proxy_score']}",
            f"- 通信指标说明：{report['evidence_notes']['communication_reduction']}",
            f"- 计算指标说明：{report['evidence_notes']['computation_overhead']}",
        ]
        return "\n".join(lines) + "\n"
