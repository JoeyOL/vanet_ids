import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from config import PathConfig, TrainingConfig
from federated_learning import FedAvgProxOptimizer
from schemas import ClientState, PreprocessResult


class FedAvgProxOptimizerTestCase(unittest.TestCase):
    def test_select_clients_prefers_high_score_clients_and_applies_fairness_penalty(self) -> None:
        client_states = [
            ClientState(client_id=0, compute_capacity=1.0, battery_level=1.0, channel_quality=1.0),
            ClientState(client_id=1, compute_capacity=0.9, battery_level=0.9, channel_quality=0.9),
            ClientState(client_id=2, compute_capacity=0.7, battery_level=0.7, channel_quality=0.7),
            ClientState(client_id=3, compute_capacity=0.6, battery_level=0.6, channel_quality=0.6),
            ClientState(client_id=4, compute_capacity=0.5, battery_level=0.5, channel_quality=0.5),
        ]
        optimizer = FedAvgProxOptimizer(
            training_config=TrainingConfig(global_rounds=3, num_clients=5, client_fraction=0.4),
            client_states=client_states,
        )

        round_one = [client.client_id for client in optimizer.select_clients(round_index=0)]
        client_states[0].metadata["times_selected"] = 4
        round_two = [client.client_id for client in optimizer.select_clients(round_index=1)]

        self.assertEqual(round_one, [0, 1])
        self.assertEqual(round_two, [1, 0])
        self.assertLess(client_states[0].last_selection_score, 1.0)
        self.assertLess(client_states[0].last_selection_score, client_states[1].last_selection_score)

    def test_aggregate_weights_uses_sample_counts(self) -> None:
        optimizer = FedAvgProxOptimizer(training_config=TrainingConfig(num_clients=2, global_rounds=1))
        client_updates = [
            {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([1.0])},
            {"weight": torch.tensor([5.0, 7.0]), "bias": torch.tensor([3.0])},
        ]
        selected_clients = [ClientState(client_id=0, num_samples=1), ClientState(client_id=1, num_samples=3)]

        aggregated = optimizer.aggregate_weights(client_updates, selected_clients)

        self.assertTrue(torch.allclose(aggregated["weight"], torch.tensor([4.0, 6.0])))
        self.assertTrue(torch.allclose(aggregated["bias"], torch.tensor([2.5])))

    def test_compress_client_update_is_identity_when_compression_disabled(self) -> None:
        optimizer = FedAvgProxOptimizer(
            training_config=TrainingConfig(
                num_clients=1,
                global_rounds=1,
                compression_topk_ratio=1.0,
                quantization_bits=32,
            )
        )
        global_state = {"weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        updated_state = {"weight": torch.tensor([1.5, 2.5], dtype=torch.float32)}

        reconstructed_state, compression_stats = optimizer._compress_client_update(updated_state, global_state)

        self.assertTrue(torch.allclose(reconstructed_state["weight"], updated_state["weight"]))
        self.assertEqual(compression_stats["original_bytes"], compression_stats["compressed_bytes"])
        self.assertEqual(compression_stats["reduction_ratio"], 0.0)

    def test_train_respects_round_count_and_writes_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output_dir = base / "outputs"
            path_config = PathConfig(data_dir=base / "data", output_dir=output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            train_dataset = pd.DataFrame(
                {
                    "feature_1": [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0],
                    "feature_2": [1.0, 1.1, 1.2, 1.3, 1.6, 1.7, 1.8, 1.9],
                    "feature_3": [2.0, 2.1, 2.2, 2.3, 2.6, 2.7, 2.8, 2.9],
                    "feature_4": [3.0, 3.1, 3.2, 3.3, 3.6, 3.7, 3.8, 3.9],
                    "label": [0, 0, 0, 0, 1, 1, 1, 1],
                }
            )
            validation_dataset = pd.DataFrame(
                {
                    "feature_1": [0.15, 0.35, 0.75, 0.95],
                    "feature_2": [1.05, 1.25, 1.65, 1.85],
                    "feature_3": [2.05, 2.25, 2.65, 2.85],
                    "feature_4": [3.05, 3.25, 3.65, 3.85],
                    "label": [0, 0, 1, 1],
                }
            )
            train_dataset.to_csv(path_config.preprocessed_train_path, index=False)
            validation_dataset.to_csv(path_config.preprocessed_val_path, index=False)
            with path_config.label_mapping_path.open("w", encoding="utf-8") as file:
                json.dump({"benign": 0, "attack": 1}, file)

            preprocess_result = PreprocessResult(
                train_dataset_path=path_config.preprocessed_train_path,
                validation_dataset_path=path_config.preprocessed_val_path,
                label_mapping_path=path_config.label_mapping_path,
                feature_columns=["feature_1", "feature_2", "feature_3", "feature_4"],
                label_column="label",
                train_size=8,
                validation_size=4,
                num_classes=2,
            )
            optimizer = FedAvgProxOptimizer(
                training_config=TrainingConfig(
                    global_rounds=2,
                    num_clients=2,
                    local_epochs=1,
                    batch_size=2,
                    learning_rate=0.01,
                    client_fraction=1.0,
                ),
                path_config=path_config,
            )

            history = optimizer.train(preprocess_result=preprocess_result)

            self.assertEqual(len(history), 2)
            self.assertEqual([record["round"] for record in history], [1, 2])
            self.assertTrue(path_config.training_history_path.exists())
            self.assertTrue(path_config.resolve_checkpoint_path("global_model.pt").exists())
            self.assertTrue(path_config.federated_report_path.exists())
            self.assertTrue(path_config.federated_report_markdown_path.exists())
            with path_config.training_history_path.open("r", encoding="utf-8") as file:
                persisted_history = json.load(file)
            with path_config.federated_report_path.open("r", encoding="utf-8") as file:
                federated_report = json.load(file)
            self.assertEqual(len(persisted_history), 2)
            self.assertIn("performance_summary", federated_report)
            self.assertIn("overall_communication_reduction_ratio", federated_report["performance_summary"])


if __name__ == "__main__":
    unittest.main()
