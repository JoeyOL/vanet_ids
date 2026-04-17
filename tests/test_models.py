import tempfile
import unittest
from pathlib import Path

import torch

from config import PathConfig, TrainingConfig
from metrics import calculate_classification_metrics
from models import LightweightCNNLSTM


class LightweightCNNLSTMTestCase(unittest.TestCase):
    def test_forward_output_shape(self) -> None:
        model = LightweightCNNLSTM(input_dim=12, num_classes=3, training_config=TrainingConfig(batch_size=4))
        inputs = torch.randn(4, 12)

        logits = model(inputs)

        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_checkpoint_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            path_config = PathConfig(data_dir=base / "data", output_dir=base / "outputs")
            model = LightweightCNNLSTM(input_dim=8, num_classes=2, path_config=path_config)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(0.25)

            checkpoint_path = model.save_checkpoint()
            reloaded_model = LightweightCNNLSTM(input_dim=8, num_classes=2, path_config=path_config)
            reloaded_model.load_checkpoint(checkpoint_path)

            for original, reloaded in zip(model.state_dict().values(), reloaded_model.state_dict().values()):
                self.assertTrue(torch.allclose(original, reloaded))

    def test_metric_calculation(self) -> None:
        result = calculate_classification_metrics(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            loss=0.2,
            latency_ms=12.5,
        )

        self.assertAlmostEqual(result.accuracy, 0.75)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.false_positive_rate, 0.5)
        self.assertAlmostEqual(result.precision, 2 / 3)
        self.assertAlmostEqual(result.f1_score, 0.8)
        self.assertEqual(result.sample_count, 4)


if __name__ == "__main__":
    unittest.main()
