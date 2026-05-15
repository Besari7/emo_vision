from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS
from multimodal_emotion.inference.fusion import FusionEngine
from multimodal_emotion.inference.result import PredictionResult


def result(modality: str, probs: list[float], quality: float = 1.0) -> PredictionResult:
    return PredictionResult(
        modality=modality,  # type: ignore[arg-type]
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=[0.0] * 7,
        probs=probs,
        pred_label=CANONICAL_LABELS[max(range(7), key=lambda idx: probs[idx])],
        confidence=max(probs),
        quality={"quality_weight_multiplier": quality},
    )


def result_with_confidence(modality: str, probs: list[float], confidence: float, quality: float = 1.0) -> PredictionResult:
    return PredictionResult(
        modality=modality,  # type: ignore[arg-type]
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=[0.0] * 7,
        probs=probs,
        pred_label=CANONICAL_LABELS[max(range(7), key=lambda idx: probs[idx])],
        confidence=confidence,
        quality={"quality_weight_multiplier": quality},
    )


class FusionTest(unittest.TestCase):
    def test_missing_modality_renormalizes_weights(self) -> None:
        engine = FusionEngine()
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        audio = PredictionResult.from_unavailable("audio", "missing")
        video = result("video", [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])

        fused = engine.fuse([text, audio, video])
        self.assertTrue(fused.available)
        self.assertIsNotNone(fused.probs)
        self.assertAlmostEqual(sum(fused.probs or []), 1.0, places=6)
        self.assertEqual(set(fused.quality["weights_used"].keys()), {"text", "video"})
        self.assertAlmostEqual(sum(fused.quality["weights_used"].values()), 1.0, places=6)

    def test_single_modality_returns_same_distribution(self) -> None:
        engine = FusionEngine()
        text_probs = [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1]
        fused = engine.fuse([result("text", text_probs)])
        self.assertEqual(fused.modality, "fusion")
        self.assertEqual(fused.pred_label, "joy")
        self.assertEqual(fused.probs, text_probs)
        self.assertEqual(fused.quality["weights_used"], {"text": 1.0})

    def test_zero_quality_skips_modality(self) -> None:
        engine = FusionEngine()
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], quality=0.0)
        video = result("video", [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])
        fused = engine.fuse([text, video])
        self.assertEqual(fused.quality["weights_used"], {"video": 1.0})
        self.assertEqual(fused.pred_label, "joy")

    def test_log_probs_probability_sum(self) -> None:
        engine = FusionEngine(mode="log_probs")
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        audio = result("audio", [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1])
        fused = engine.fuse([text, audio])
        self.assertEqual(fused.quality["mode"], "log_probs")
        self.assertIsNotNone(fused.probs)
        self.assertAlmostEqual(sum(fused.probs or []), 1.0, places=6)

    def test_log_probs_missing_modality_renormalizes_weights(self) -> None:
        engine = FusionEngine(mode="log_probs")
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        audio = PredictionResult.from_unavailable("audio", "missing")
        video = result("video", [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])
        fused = engine.fuse([text, audio, video])
        self.assertEqual(fused.quality["mode"], "log_probs")
        self.assertEqual(set(fused.quality["weights_used"].keys()), {"text", "video"})
        self.assertAlmostEqual(sum(fused.quality["weights_used"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(fused.probs or []), 1.0, places=6)

    def test_confidence_gating_disabled_does_not_change_weights(self) -> None:
        engine = FusionEngine(global_weights={"text": 0.5, "audio": 0.0, "video": 0.5})
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        video = result_with_confidence(
            "video",
            [0.20, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13],
            confidence=0.20,
        )
        fused = engine.fuse([text, video])
        self.assertEqual(set(fused.quality["weights_used"].keys()), {"text", "video"})
        self.assertAlmostEqual(fused.quality["weights_used"]["text"], 0.5, places=6)
        self.assertAlmostEqual(fused.quality["weights_used"]["video"], 0.5, places=6)

    def test_confidence_gating_enabled_zeroes_low_confidence_video(self) -> None:
        engine = FusionEngine(
            global_weights={"text": 0.5, "audio": 0.0, "video": 0.5},
            confidence_gating={
                "enabled": True,
                "video_min_conf_zero": 0.25,
                "video_min_conf_low": 0.35,
                "video_low_multiplier": 0.25,
            },
        )
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        video = result_with_confidence(
            "video",
            [0.20, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13],
            confidence=0.20,
        )
        fused = engine.fuse([text, video])
        self.assertEqual(fused.quality["weights_used"], {"text": 1.0})

    def test_confidence_gating_enabled_lowers_borderline_video_weight(self) -> None:
        engine = FusionEngine(
            global_weights={"text": 0.5, "audio": 0.0, "video": 0.5},
            confidence_gating={
                "enabled": True,
                "video_min_conf_zero": 0.25,
                "video_min_conf_low": 0.35,
                "video_low_multiplier": 0.25,
            },
        )
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        video = result_with_confidence(
            "video",
            [0.30, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11],
            confidence=0.30,
        )
        fused = engine.fuse([text, video])
        self.assertLessEqual(fused.quality["weights_used"]["video"], 0.25)
        self.assertAlmostEqual(sum(fused.quality["weights_used"].values()), 1.0, places=6)

    def test_zero_global_weight_modality_is_ignored(self) -> None:
        engine = FusionEngine(global_weights={"text": 1.0, "audio": 0.0, "video": 0.0})
        text = result("text", [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        audio = result("audio", [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1])
        video = result("video", [0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1])
        fused = engine.fuse([text, audio, video])
        self.assertEqual(fused.quality["weights_used"], {"text": 1.0})


if __name__ == "__main__":
    unittest.main()
