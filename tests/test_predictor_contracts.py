from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS
from multimodal_emotion.inference.result import PredictionResult


class PredictionResultContractTest(unittest.TestCase):
    def test_to_dict_schema(self) -> None:
        result = PredictionResult(
            modality="text",
            available=True,
            labels=list(CANONICAL_LABELS),
            logits=[0.0] * 7,
            probs=[1.0 / 7.0] * 7,
            pred_label="neutral",
            confidence=1.0 / 7.0,
            quality={"quality_weight_multiplier": 1.0},
        )
        payload = result.to_dict()
        self.assertEqual(payload["modality"], "text")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["labels"], CANONICAL_LABELS)
        self.assertEqual(len(payload["logits"]), 7)
        self.assertEqual(len(payload["probs"]), 7)
        self.assertEqual(payload["pred_label"], "neutral")
        self.assertIn("quality_weight_multiplier", payload["quality"])

    def test_unavailable_result_shape(self) -> None:
        result = PredictionResult.from_unavailable("audio", "decode failed", {"duration_sec": 0.0})
        payload = result.to_dict()
        self.assertFalse(payload["available"])
        self.assertIsNone(payload["logits"])
        self.assertIsNone(payload["probs"])
        self.assertIsNone(payload["pred_label"])
        self.assertEqual(payload["error"], "decode failed")
        self.assertEqual(payload["quality"]["quality_weight_multiplier"], 0.0)

    def test_rejects_noncanonical_result_labels(self) -> None:
        with self.assertRaises(ValueError):
            PredictionResult(
                modality="video",
                available=True,
                labels=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                probs=[1.0 / 7.0] * 7,
            )


if __name__ == "__main__":
    unittest.main()
