import json
import tempfile
import unittest
from pathlib import Path

from config import DatasetConfig, PathConfig, TrainingConfig, build_app_config, load_config_file


class ConfigTestCase(unittest.TestCase):
    def test_default_configs(self) -> None:
        dataset_config = DatasetConfig()
        path_config = PathConfig()
        training_config = TrainingConfig()

        self.assertEqual(dataset_config.dataset_name, "VeReMi & Car-Hacking")
        self.assertEqual(dataset_config.label_column, "label")
        self.assertEqual(training_config.global_rounds, 50)
        self.assertEqual(training_config.num_clients, 10)
        self.assertEqual(training_config.batch_size, 32)
        self.assertEqual(training_config.learning_rate, 1e-3)
        self.assertEqual(training_config.device, "cpu")
        self.assertTrue(path_config.data_dir.is_absolute())
        self.assertTrue(path_config.output_dir.is_absolute())

    def test_parameter_override(self) -> None:
        dataset_config = DatasetConfig(dataset_name="Custom Dataset", label_column="attack_type")
        path_config = PathConfig(
            data_dir="./fixtures/data",
            output_dir="./fixtures/output",
            checkpoint_path="./fixtures/checkpoint/model.pt",
        )
        training_config = TrainingConfig(
            global_rounds=3,
            num_clients=4,
            local_epochs=2,
            batch_size=8,
            learning_rate=0.01,
            device="cpu",
            client_fraction=0.75,
            fedprox_mu=0.1,
        )

        self.assertEqual(dataset_config.dataset_name, "Custom Dataset")
        self.assertEqual(dataset_config.label_column, "attack_type")
        self.assertEqual(training_config.global_rounds, 3)
        self.assertEqual(training_config.num_clients, 4)
        self.assertEqual(training_config.local_epochs, 2)
        self.assertEqual(training_config.batch_size, 8)
        self.assertEqual(training_config.learning_rate, 0.01)
        self.assertEqual(training_config.client_fraction, 0.75)
        self.assertEqual(training_config.fedprox_mu, 0.1)
        self.assertEqual(path_config.checkpoint_path.name, "model.pt")

    def test_path_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            raw_data_dir = base / "datasets" / ".." / "datasets" / "vereMi"
            raw_output_dir = base / "artifacts" / "." / "results"
            raw_checkpoint = base / "artifacts" / "results" / ".." / "global.pt"

            path_config = PathConfig(
                data_dir=raw_data_dir,
                output_dir=raw_output_dir,
                checkpoint_path=raw_checkpoint,
            )

            self.assertEqual(path_config.data_dir, raw_data_dir.resolve())
            self.assertEqual(path_config.output_dir, raw_output_dir.resolve())
            self.assertEqual(path_config.checkpoint_path, raw_checkpoint.resolve())
            self.assertEqual(
                path_config.resolve_checkpoint_path(),
                raw_checkpoint.resolve(),
            )

    def test_load_config_file_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "config.json"
            config_payload = {
                "path": {
                    "data_dir": "./data",
                    "output_dir": "./outputs",
                    "checkpoint_path": "./outputs/model.pt",
                },
                "training": {"global_rounds": 2},
            }
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")

            loaded_payload = load_config_file(config_path)
            app_config = build_app_config(loaded_payload)

            self.assertEqual(app_config.path.data_dir, (base / "data").resolve())
            self.assertEqual(app_config.path.output_dir, (base / "outputs").resolve())
            self.assertEqual(app_config.path.checkpoint_path, (base / "outputs" / "model.pt").resolve())
            self.assertEqual(app_config.training.global_rounds, 2)


if __name__ == "__main__":
    unittest.main()
