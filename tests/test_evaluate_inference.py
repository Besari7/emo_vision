from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS
from multimodal_emotion.inference.result import PredictionResult
from scripts.evaluate_inference import collapse_warnings_from_prediction_rows, evaluate_rows, read_manifest


class FakePredictor:
    def __init__(self, modality: str, pred_idx: int) -> None:
        self.modality = modality
        self.pred_idx = pred_idx

    def predict(self, _value):
        probs = [0.05] * len(CANONICAL_LABELS)
        probs[self.pred_idx] = 0.70
        total = sum(probs)
        probs = [value / total for value in probs]
        return PredictionResult(
            modality=self.modality,  # type: ignore[arg-type]
            available=True,
            labels=list(CANONICAL_LABELS),
            logits=[0.0] * len(CANONICAL_LABELS),
            probs=probs,
            pred_label=CANONICAL_LABELS[self.pred_idx],
            confidence=probs[self.pred_idx],
            quality={"quality_weight_multiplier": 1.0},
        )


class EvaluateInferenceTest(unittest.TestCase):
    def test_evaluate_inference_writes_expected_outputs(self) -> None:
        temp_root = ROOT / "artifacts" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / f"eval_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "s1",
                        "label": "joy",
                        "text": "happy",
                        "audio_path": "missing.wav",
                        "video_path": "missing.mp4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = read_manifest(manifest)
            output_dir = root / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = evaluate_rows(
                    rows,
                    modality="fusion",
                    output_dir=output_dir,
                    fusion_mode="log_probs",
                    predictors={
                        "text": FakePredictor("text", CANONICAL_LABELS.index("joy")),
                        "audio": FakePredictor("audio", CANONICAL_LABELS.index("joy")),
                        "video": FakePredictor("video", CANONICAL_LABELS.index("joy")),
                    },
                )

            self.assertEqual(metrics["primary_metric"], "macro_f1")
            self.assertEqual(metrics["fusion_mode"], "log_probs")
            for filename in [
                "metrics.json",
                "classification_report.txt",
                "confusion_matrix.csv",
                "predictions.csv",
                "debug_predictions.jsonl",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            debug_payload = json.loads((output_dir / "debug_predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(debug_payload["fusion_debug"]["mode"], "log_probs")
            self.assertEqual(debug_payload["fusion"]["labels"], CANONICAL_LABELS)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_collapse_warning_detects_one_label_dominance(self) -> None:
        rows = [
            {"pred_label": "fear"},
            {"pred_label": "fear"},
            {"pred_label": "fear"},
            {"pred_label": "joy"},
        ]
        warnings = collapse_warnings_from_prediction_rows(rows, threshold=0.70)
        self.assertEqual(warnings, ["WARNING: predictions collapsed: fear = 3/4"])


if __name__ == "__main__":
    unittest.main()
