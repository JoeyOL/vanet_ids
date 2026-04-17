import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import DatasetConfig, PathConfig
from data_processor import DataProcessor


class DataProcessorTestCase(unittest.TestCase):
    def test_process_local_data_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_dir = base / "data"
            output_dir = base / "outputs"
            data_dir.mkdir()

            dataset = pd.DataFrame(
                {
                    "Timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
                    "CAN_ID": [100, 101, 102, 103, 104, 105, 106, 107],
                    "Speed": [35.0, 36.5, None, 38.5, 39.0, 40.2, 42.1, None],
                    "Payload": [
                        "AA 00 FF",
                        "BB CC DD",
                        None,
                        "00 00 10",
                        "1A 2B 3C",
                        "FF FF FF",
                        "10 20 30",
                        "AB CD EF",
                    ],
                    "Protocol": ["CAN", "CAN", "V2X", "CAN", "V2X", "CAN", "V2X", "CAN"],
                    "Label": [
                        "Normal",
                        "DoS Attack",
                        "Normal",
                        "Spoofing",
                        "Normal",
                        "DoS Attack",
                        "Normal",
                        "Spoofing",
                    ],
                }
            )
            dataset.to_csv(data_dir / "vehicular.csv", index=False)

            processor = DataProcessor(
                dataset_config=DatasetConfig(validation_split=0.25),
                path_config=PathConfig(data_dir=data_dir, output_dir=output_dir),
            )

            result = processor.process_local_data()

            self.assertTrue(result.train_dataset_path.exists())
            self.assertTrue(result.validation_dataset_path.exists())
            self.assertTrue(result.label_mapping_path.exists())
            self.assertEqual(result.train_size + result.validation_size, 8)
            self.assertGreater(result.num_classes, 1)
            self.assertIn("speed", result.feature_columns)
            self.assertIn("payload_hex_length", result.feature_columns)

            train_dataset = pd.read_csv(result.train_dataset_path)
            validation_dataset = pd.read_csv(result.validation_dataset_path)
            with result.label_mapping_path.open("r", encoding="utf-8") as file:
                label_mapping = json.load(file)

            self.assertIn("label", train_dataset.columns)
            self.assertIn("label", validation_dataset.columns)
            self.assertEqual(int(train_dataset.isna().sum().sum()), 0)
            self.assertEqual(int(validation_dataset.isna().sum().sum()), 0)
            self.assertIn("benign", label_mapping)
            self.assertIn("dos_attack", label_mapping)
            self.assertIn("spoofing", label_mapping)

    def test_process_local_data_raises_for_missing_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_dir = base / "data"
            output_dir = base / "outputs"
            data_dir.mkdir()

            dataset = pd.DataFrame(
                {
                    "timestamp": [1, 2, 3],
                    "can_id": [100, 101, 102],
                    "payload": ["AA", "BB", "CC"],
                }
            )
            dataset.to_csv(data_dir / "broken.csv", index=False)

            processor = DataProcessor(path_config=PathConfig(data_dir=data_dir, output_dir=output_dir))

            with self.assertRaises(ValueError):
                processor.process_local_data()

    def test_process_local_data_raises_for_missing_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            processor = DataProcessor(
                path_config=PathConfig(data_dir=base / "data", output_dir=base / "outputs")
            )

            with self.assertRaises(FileNotFoundError):
                processor.process_local_data()


if __name__ == "__main__":
    unittest.main()
