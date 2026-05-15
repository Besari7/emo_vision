from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import (
    CANONICAL_LABELS,
    label_to_id,
    normalize_label,
    reorder_logits_to_canonical,
    reorder_probs_to_canonical,
    validate_label_set,
)


class LabelUtilitiesTest(unittest.TestCase):
    def test_canonical_label_order(self) -> None:
        self.assertEqual(
            CANONICAL_LABELS,
            ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
        )
        self.assertEqual(label_to_id["neutral"], 0)
        self.assertEqual(label_to_id["anger"], 6)

    def test_alias_mapping(self) -> None:
        self.assertEqual(normalize_label("happy"), "joy")
        self.assertEqual(normalize_label("sad"), "sadness")
        self.assertEqual(normalize_label("angry"), "anger")

    def test_validate_label_set_rejects_missing_unknown_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            validate_label_set(["neutral", "surprise"])
        with self.assertRaises(ValueError):
            validate_label_set(["neutral", "surprise", "fear", "sadness", "joy", "disgust", "other"])
        with self.assertRaises(ValueError):
            validate_label_set(["neutral", "surprise", "fear", "sadness", "joy", "anger", "angry"])

    def test_reorder_audio_video_order_to_canonical(self) -> None:
        source_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        logits = np.array([6, 5, 2, 4, 0, 3, 1], dtype=np.float64)
        aligned_logits = reorder_logits_to_canonical(logits, source_labels)
        np.testing.assert_array_equal(aligned_logits, np.arange(7, dtype=np.float64))

        probs = np.array([0.07, 0.06, 0.03, 0.05, 0.01, 0.04, 0.02], dtype=np.float64)
        aligned_probs = reorder_probs_to_canonical(probs, source_labels)
        np.testing.assert_array_equal(aligned_probs, np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]))


if __name__ == "__main__":
    unittest.main()
