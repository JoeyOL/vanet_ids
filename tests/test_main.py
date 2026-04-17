import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import main as app_main


class MainEntryTestCase(unittest.TestCase):
    def test_build_parser_parses_extended_options(self) -> None:
        parser = app_main.build_parser()

        args = parser.parse_args(
            [
                "--mode",
                "train",
                "--data-dir",
                "./demo-data",
                "--output-dir",
                "./demo-output",
                "--batch-size",
                "16",
                "--epochs",
                "2",
                "--lr",
                "0.005",
                "--device",
                "cpu",
                "--checkpoint",
                "./demo-output/model.pt",
                "--topk-ratio",
                "0.2",
                "--quant-bits",
                "16",
                "--selection-weight-compute",
                "0.5",
                "--selection-weight-battery",
                "0.3",
                "--selection-weight-channel",
                "0.2",
            ]
        )

        self.assertEqual(args.mode, "train")
        self.assertEqual(args.data_dir, "./demo-data")
        self.assertEqual(args.output_dir, "./demo-output")
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.lr, 0.005)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.checkpoint, "./demo-output/model.pt")
        self.assertEqual(args.topk_ratio, 0.2)
        self.assertEqual(args.quant_bits, 16)
        self.assertEqual(args.selection_weight_compute, 0.5)
        self.assertEqual(args.selection_weight_battery, 0.3)
        self.assertEqual(args.selection_weight_channel, 0.2)

    def test_main_preprocess_mode_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_dir = base / "raw_data"
            output_dir = base / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_dataset(data_dir / "vehicular.csv")

            exit_code = app_main.main(
                [
                    "--mode",
                    "preprocess",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "train_processed.csv").exists())
            self.assertTrue((output_dir / "val_processed.csv").exists())
            self.assertTrue((output_dir / "label_mapping.json").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())
            self.assertTrue((output_dir / "config_snapshot.json").exists())
            self.assertTrue((output_dir / "logs" / "app.log").exists())

    def test_main_train_mode_runs_with_preprocessed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_processed_artifacts(output_dir)

            exit_code = app_main.main(
                [
                    "--mode",
                    "train",
                    "--data-dir",
                    str(base / "unused-data"),
                    "--output-dir",
                    str(output_dir),
                    "--rounds",
                    "1",
                    "--clients",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "training_history.json").exists())
            self.assertTrue((output_dir / "global_model.pt").exists())
            self.assertTrue((output_dir / "reports" / "federated_training_report.json").exists())
            self.assertTrue((output_dir / "reports" / "federated_training_report.md").exists())

    def test_main_test_mode_returns_error_when_checkpoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_processed_artifacts(output_dir)

            exit_code = app_main.main(
                [
                    "--mode",
                    "test",
                    "--data-dir",
                    str(base / "unused-data"),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(exit_code, 1)

    def test_main_supports_config_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_dir = base / "raw_data"
            output_dir = base / "outputs"
            config_dir = base / "configs"
            data_dir.mkdir(parents=True, exist_ok=True)
            config_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_dataset(data_dir / "vehicular.csv")

            config_path = config_dir / "default.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[dataset]",
                        'dataset_name = "Demo Dataset"',
                        'label_column = "label"',
                        "validation_split = 0.2",
                        "",
                        "[path]",
                        'data_dir = "../raw_data"',
                        'output_dir = "../outputs"',
                        'checkpoint_path = "../outputs/global_model.pt"',
                        "",
                        "[training]",
                        "global_rounds = 1",
                        "num_clients = 2",
                        "local_epochs = 1",
                        "batch_size = 2",
                        "learning_rate = 0.001",
                        'device = "cpu"',
                        "client_fraction = 1.0",
                        "fedprox_mu = 0.0",
                        "compression_topk_ratio = 0.2",
                        "quantization_bits = 16",
                        "selection_weight_compute = 0.5",
                        "selection_weight_battery = 0.3",
                        "selection_weight_channel = 0.2",
                        "",
                        "[logging]",
                        'level = "INFO"',
                        "json_logs = false",
                        "log_to_file = true",
                        "",
                        "[runtime]",
                        "seed = 7",
                        "deterministic = true",
                        'run_name = "from-config"',
                    ]
                ),
                encoding="utf-8",
            )

            exit_code = app_main.main(["--mode", "preprocess", "--config", str(config_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "train_processed.csv").exists())
            self.assertTrue((output_dir / "logs" / "app.log").exists())

    def test_build_configs_uses_output_dir_checkpoint_when_output_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "default.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[path]",
                        'data_dir = "./data"',
                        'output_dir = "./outputs"',
                        'checkpoint_path = "./outputs/from-config.pt"',
                    ]
                ),
                encoding="utf-8",
            )

            argv = [
                "--config",
                str(config_path),
                "--output-dir",
                str(base / "custom-output"),
            ]
            parser = app_main.build_parser(app_main.load_parser_defaults(argv))
            args = parser.parse_args(argv)

            app_config = app_main.build_configs(args, explicit_output_dir=True, explicit_checkpoint=False)

            self.assertEqual(
                app_config.path.resolve_checkpoint_path("global_model.pt"),
                (base / "custom-output" / "global_model.pt").resolve(),
            )

    def _write_raw_dataset(self, file_path: Path) -> None:
        raw_dataset = pd.DataFrame(
            {
                "Timestamp": [1, 2, 3, 4, 5, 6],
                "CAN_ID": [100, 101, 102, 103, 104, 105],
                "Speed": [30.0, 31.5, 32.0, None, 35.1, 36.4],
                "Payload": ["AA 00", "BB 01", "CC 02", "DD 03", None, "FF 05"],
                "Protocol": ["CAN", "CAN", "V2X", "CAN", "V2X", "CAN"],
                "Label": ["Normal", "DoS Attack", "Normal", "Spoofing", "Normal", "DoS Attack"],
            }
        )
        raw_dataset.to_csv(file_path, index=False)

    def _write_processed_artifacts(self, output_dir: Path) -> None:
        train_dataset = pd.DataFrame(
            {
                "feature_1": [0.1, 0.2, 0.8, 0.9],
                "feature_2": [1.0, 1.1, 1.8, 1.9],
                "feature_3": [2.0, 2.1, 2.8, 2.9],
                "feature_4": [3.0, 3.1, 3.8, 3.9],
                "label": [0, 0, 1, 1],
            }
        )
        validation_dataset = pd.DataFrame(
            {
                "feature_1": [0.15, 0.95],
                "feature_2": [1.05, 1.95],
                "feature_3": [2.05, 2.95],
                "feature_4": [3.05, 3.95],
                "label": [0, 1],
            }
        )
        train_dataset.to_csv(output_dir / "train_processed.csv", index=False)
        validation_dataset.to_csv(output_dir / "val_processed.csv", index=False)
        with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
            json.dump({"benign": 0, "attack": 1}, file)


if __name__ == "__main__":
    unittest.main()
