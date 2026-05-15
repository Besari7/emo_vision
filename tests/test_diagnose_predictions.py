from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_predictions import build_diagnostics, collapse_warnings_from_prediction_rows


class DiagnosePredictionsTest(unittest.TestCase):
    def test_collapse_detection_warns_on_dominant_label(self) -> None:
        rows = [
            {"sample_id": "1", "true_label": "joy", "pred_label": "fear", "confidence": "0.8"},
            {"sample_id": "2", "true_label": "fear", "pred_label": "fear", "confidence": "0.7"},
            {"sample_id": "3", "true_label": "sadness", "pred_label": "fear", "confidence": "0.6"},
            {"sample_id": "4", "true_label": "neutral", "pred_label": "fear", "confidence": "0.5"},
            {"sample_id": "5", "true_label": "joy", "pred_label": "joy", "confidence": "0.9"},
        ]
        warnings = collapse_warnings_from_prediction_rows(rows, threshold=0.79)
        self.assertEqual(warnings, ["WARNING: predictions collapsed: fear = 4/5"])

    def test_diagnostics_include_metrics_and_quality_summaries(self) -> None:
        prediction_rows = [
            {"sample_id": "1", "true_label": "joy", "pred_label": "joy", "confidence": "0.8"},
            {"sample_id": "2", "true_label": "fear", "pred_label": "joy", "confidence": "0.6"},
        ]
        debug_rows = [
            {
                "audio": {
                    "available": True,
                    "quality": {"duration_sec": 2.0, "rms": 0.1, "num_chunks": 1, "quality_weight_multiplier": 1.0},
                },
                "video": {
                    "available": False,
                    "quality": {
                        "face_ratio": 0.5,
                        "face_frames": 8,
                        "sampled_frames": 16,
                        "quality_weight_multiplier": 0.5,
                    },
                },
                "fusion": {"available": True, "quality": {}},
            }
        ]
        diagnostics = build_diagnostics(prediction_rows, debug_rows)
        self.assertEqual(diagnostics["accuracy"], 0.5)
        self.assertIn("macro_f1", diagnostics)
        self.assertEqual(diagnostics["unavailable_count"]["video"], 1)
        self.assertEqual(diagnostics["audio_quality"]["quality_weight_multiplier"]["count"], 1)
        self.assertEqual(diagnostics["video_quality"]["face_frames"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
