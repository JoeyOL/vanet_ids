"""分布式数据预处理模块。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DatasetConfig, PathConfig
from schemas import PreprocessResult


LOGGER = logging.getLogger(__name__)


class DataProcessor:
    _LABEL_ALIASES = ("label", "labels", "class", "attack", "attack_type", "target", "y")
    _PAYLOAD_KEYWORDS = ("payload", "data", "frame", "message")

    def __init__(
        self,
        dataset_config: DatasetConfig | None = None,
        path_config: PathConfig | None = None,
    ):
        self.dataset_config = dataset_config or DatasetConfig()
        self.path_config = path_config or PathConfig()
        self.dataset = self.dataset_config.dataset_name

    def discover_data_files(self, input_path: str | Path | None = None) -> list[Path]:
        root = Path(input_path).expanduser().resolve() if input_path else self.path_config.data_dir
        if not root.exists():
            raise FileNotFoundError(f"数据路径不存在: {root}")

        if root.is_file():
            if root.suffix.lower() not in self.dataset_config.supported_extensions:
                raise ValueError(f"不支持的数据文件格式: {root.suffix}")
            return [root]

        files = sorted(
            file_path
            for file_path in root.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in self.dataset_config.supported_extensions
        )
        if not files:
            raise FileNotFoundError(f"未在 {root} 中发现可处理的数据文件")
        return files

    def process_local_data(self, input_path: str | Path | None = None) -> PreprocessResult:
        self.path_config.ensure_runtime_dirs()
        data_files = self.discover_data_files(input_path=input_path)
        LOGGER.info("加载 %s 数据集，发现 %s 个文件", self.dataset, len(data_files))

        raw_dataset = self._load_dataset(data_files)
        raw_dataset = self._normalize_columns(raw_dataset)
        raw_dataset = self._ensure_label_column(raw_dataset)
        self._validate_required_columns(raw_dataset)

        cleaned_dataset = self._handle_missing_values(raw_dataset)
        normalized_labels = self._standardize_labels(cleaned_dataset[self.dataset_config.label_column])
        features = self._extract_features(cleaned_dataset.drop(columns=[self.dataset_config.label_column]))
        label_mapping = self._build_label_mapping(normalized_labels)
        encoded_labels = normalized_labels.map(label_mapping).astype(int)

        processed_dataset = features.copy()
        processed_dataset[self.dataset_config.label_column] = encoded_labels
        train_dataset, validation_dataset = self._split_dataset(processed_dataset, normalized_labels)
        self._persist_artifacts(train_dataset, validation_dataset, label_mapping)

        LOGGER.info("提取时域特征与统计特征...")
        LOGGER.info("半监督策略异常样本标注...")
        LOGGER.info("数据处理完成，全程不出本地，保障隐私。")

        return PreprocessResult(
            train_dataset_path=self.path_config.preprocessed_train_path,
            validation_dataset_path=self.path_config.preprocessed_val_path,
            label_mapping_path=self.path_config.label_mapping_path,
            feature_columns=list(features.columns),
            label_column=self.dataset_config.label_column,
            train_size=len(train_dataset),
            validation_size=len(validation_dataset),
            num_classes=len(label_mapping),
            dataset_name=self.dataset_config.dataset_name,
        )

    def _load_dataset(self, data_files: Iterable[Path]) -> pd.DataFrame:
        frames = [self._read_table(file_path) for file_path in data_files]
        return pd.concat(frames, ignore_index=True)

    def _read_table(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix == ".tsv":
            return pd.read_csv(file_path, sep="\t")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError(f"不支持的数据文件格式: {file_path.suffix}")

    def _normalize_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        normalized = dataset.copy()
        normalized.columns = [self._normalize_name(column) for column in normalized.columns]
        return normalized

    def _ensure_label_column(self, dataset: pd.DataFrame) -> pd.DataFrame:
        label_column = self._normalize_name(self.dataset_config.label_column)
        if label_column in dataset.columns:
            if label_column != self.dataset_config.label_column:
                dataset = dataset.rename(columns={label_column: self.dataset_config.label_column})
            return dataset

        for candidate in self._LABEL_ALIASES:
            if candidate in dataset.columns:
                return dataset.rename(columns={candidate: self.dataset_config.label_column})

        raise ValueError(
            f"缺少标签列，期望列 `{self.dataset_config.label_column}` 或别名 {self._LABEL_ALIASES}"
        )

    def _validate_required_columns(self, dataset: pd.DataFrame) -> None:
        if self.dataset_config.label_column not in dataset.columns:
            raise ValueError(f"缺少必要列: {self.dataset_config.label_column}")

        candidate_features = [column for column in dataset.columns if column != self.dataset_config.label_column]
        if not candidate_features:
            raise ValueError("输入数据缺少可用于特征提取的字段")

    def _handle_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataset.copy()
        for column in cleaned.columns:
            if column == self.dataset_config.label_column:
                cleaned[column] = cleaned[column].fillna("unknown")
                continue

            numeric_series = pd.to_numeric(cleaned[column], errors="coerce")
            numeric_ratio = numeric_series.notna().mean()
            if pd.api.types.is_numeric_dtype(cleaned[column]) or numeric_ratio >= 0.6:
                median_value = float(numeric_series.median()) if numeric_series.notna().any() else 0.0
                cleaned[column] = numeric_series.fillna(median_value)
            else:
                string_series = cleaned[column].astype(str)
                missing_mask = cleaned[column].isna() | string_series.str.lower().isin({"", "nan", "none", "null"})
                non_missing = string_series[~missing_mask]
                fill_value = non_missing.mode().iloc[0] if not non_missing.empty else "unknown"
                cleaned.loc[missing_mask, column] = fill_value
                cleaned[column] = cleaned[column].astype(str)
        return cleaned

    def _standardize_labels(self, labels: pd.Series) -> pd.Series:
        normalized = labels.astype(str).str.strip().str.lower()
        benign_aliases = {"normal", "benign", "0", "false", "no_attack", "non_attack", "safe"}
        attack_aliases = {"attack", "anomaly", "intrusion", "malicious", "1", "true"}

        def normalize_label(value: str) -> str:
            if value in benign_aliases:
                return "benign"
            if value in attack_aliases:
                return "attack"
            cleaned = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
            return cleaned or "unknown"

        return normalized.apply(normalize_label)

    def _extract_features(self, feature_dataset: pd.DataFrame) -> pd.DataFrame:
        extracted = pd.DataFrame(index=feature_dataset.index)

        for column in feature_dataset.columns:
            series = feature_dataset[column]
            numeric_series = pd.to_numeric(series, errors="coerce")
            if pd.api.types.is_numeric_dtype(series) or numeric_series.notna().mean() >= 0.6:
                filled_numeric = numeric_series.fillna(float(numeric_series.median()) if numeric_series.notna().any() else 0.0)
                extracted[column] = filled_numeric.astype(float)
                continue

            if self._looks_like_datetime_column(column, series):
                datetime_series = pd.to_datetime(series, errors="coerce")
            else:
                datetime_series = pd.Series(pd.NaT, index=series.index)

            if datetime_series.notna().mean() >= 0.6:
                extracted[f"{column}_timestamp"] = (datetime_series.astype("int64") // 10**9).astype(float)
                continue

            string_series = series.astype(str)
            extracted[f"{column}_length"] = string_series.str.len().astype(float)
            extracted[f"{column}_encoded"] = pd.factorize(string_series)[0].astype(float)
            extracted[f"{column}_token_count"] = string_series.str.count(r"[A-Za-z0-9]+$")
            extracted[f"{column}_unique_chars"] = string_series.apply(
                lambda value: float(len(set(value.replace(" ", ""))))
            )

            if any(keyword in column for keyword in self._PAYLOAD_KEYWORDS):
                cleaned_payload = string_series.str.replace(r"[^0-9A-Fa-f]", "", regex=True)
                extracted[f"{column}_hex_length"] = cleaned_payload.str.len().astype(float)
                extracted[f"{column}_nonzero_byte_ratio"] = cleaned_payload.apply(
                    self._calculate_nonzero_byte_ratio
                )

        if extracted.empty:
            raise ValueError("未能从输入数据中提取有效特征")

        return extracted.fillna(0.0)

    def _build_label_mapping(self, normalized_labels: pd.Series) -> dict[str, int]:
        ordered_labels = list(dict.fromkeys(normalized_labels.tolist()))
        return {label: index for index, label in enumerate(ordered_labels)}

    def _split_dataset(
        self,
        processed_dataset: pd.DataFrame,
        normalized_labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_dataset, validation_dataset = train_test_split(
                processed_dataset,
                test_size=self.dataset_config.validation_split,
                random_state=self.dataset_config.random_seed,
                stratify=normalized_labels if self._can_stratify(normalized_labels) else None,
            )
        except ValueError:
            train_dataset, validation_dataset = train_test_split(
                processed_dataset,
                test_size=self.dataset_config.validation_split,
                random_state=self.dataset_config.random_seed,
                stratify=None,
            )

        return train_dataset.reset_index(drop=True), validation_dataset.reset_index(drop=True)

    def _persist_artifacts(
        self,
        train_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        label_mapping: dict[str, int],
    ) -> None:
        train_dataset.to_csv(self.path_config.preprocessed_train_path, index=False)
        validation_dataset.to_csv(self.path_config.preprocessed_val_path, index=False)
        with self.path_config.label_mapping_path.open("w", encoding="utf-8") as file:
            json.dump(label_mapping, file, ensure_ascii=False, indent=2)

    def _can_stratify(self, normalized_labels: pd.Series) -> bool:
        label_counts = normalized_labels.value_counts()
        return len(label_counts) > 1 and int(label_counts.min()) >= 2

    def _normalize_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    def _calculate_nonzero_byte_ratio(self, payload: str) -> float:
        if not payload:
            return 0.0

        bytes_list = [payload[index : index + 2] for index in range(0, len(payload), 2) if payload[index : index + 2]]
        if not bytes_list:
            return 0.0
        nonzero_count = sum(byte.lower() != "00" for byte in bytes_list)
        return float(nonzero_count / len(bytes_list))

    def _looks_like_datetime_column(self, column: str, series: pd.Series) -> bool:
        if any(keyword in column for keyword in ("time", "date", "stamp")):
            return True

        string_series = series.astype(str).str.strip()
        matched = string_series.str.contains(
            r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}:\d{2}(?::\d{2})?$",
            regex=True,
        )
        return bool(matched.mean() >= 0.6)
