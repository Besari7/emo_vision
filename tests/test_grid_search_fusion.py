from __future__ import annotations

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
from scripts.grid_search_fusion import (
    copy_cache_to_output,
    grid_search,
    iter_grid,
    read_modality_cache,
    write_best_outputs,
    write_grid_results,
)


def cached_prediction(modality: str, winning_label: str) -> dict:
    logits = [0.0] * len(CANONICAL_LABELS)
    logits[CANONICAL_LABELS.index(winning_label)] = 5.0
    probs = [0.01] * len(CANONICAL_LABELS)
    probs[CANONICAL_LABELS.index(winning_label)] = 0.94
    total = sum(probs)
    probs = [value / total for value in probs]
    return PredictionResult(
        modality=modality,  # type: ignore[arg-type]
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=logits,
        probs=probs,
        pred_label=winning_label,
        confidence=max(probs),
        quality={"quality_weight_multiplier": 1.0},
    ).to_dict()


def cache_rows() -> list[dict]:
    return [
        {
            "sample_id": "s1",
            "true_label": "joy",
            "text": cached_prediction("text", "joy"),
            "audio": cached_prediction("audio", "fear"),
            "video": cached_prediction("video", "neutral"),
        },
        {
            "sample_id": "s2",
            "true_label": "sadness",
            "text": cached_prediction("text", "sadness"),
            "audio": cached_prediction("audio", "fear"),
            "video": cached_prediction("video", "neutral"),
        },
    ]


class GridSearchFusionTest(unittest.TestCase):
    def test_zero_weight_candidates_are_included(self) -> None:
        candidates = list(iter_grid("weighted_probs"))
        self.assertTrue(
            any(
                candidate["weights"] == {"text": 1.0, "audio": 0.0, "video": 0.0}
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate["weights"] == {"text": 0.0, "audio": 1.0, "video": 0.0}
                for candidate in candidates
            )
        )
        self.assertTrue(
            any(
                candidate["weights"] == {"text": 0.0, "audio": 0.0, "video": 1.0}
                for candidate in candidates
            )
        )

    def test_fake_cached_predictions_grid_search_and_text_only_candidate(self) -> None:
        results = grid_search(cache_rows(), fusion_mode="weighted_probs", metric="macro_f1")
        self.assertTrue(results)
        text_only = [
            result for result in results
            if result["weights"] == {"text": 1.0, "audio": 0.0, "video": 0.0}
        ]
        self.assertTrue(text_only)
        self.assertAlmostEqual(max(result["macro_f1"] for result in text_only), 1.0, places=6)

    def test_output_files_are_created_from_cached_predictions(self) -> None:
        temp_root = ROOT / "artifacts" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / f"grid_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        try:
            cache_path = root / "modality_cache.jsonl"
            with cache_path.open("w", encoding="utf-8") as handle:
                for row in cache_rows():
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

            rows = read_modality_cache(cache_path)
            results = grid_search(rows, fusion_mode="weighted_probs", metric="macro_f1")
            output_dir = root / "out"
            output_dir.mkdir()
            copy_cache_to_output(cache_path, output_dir / "modality_cache.jsonl")
            write_grid_results(results, output_dir / "grid_search_results.csv")
            write_best_outputs(rows, results[0], output_dir, "macro_f1")

            for filename in [
                "best_fusion_config.json",
                "grid_search_results.csv",
                "best_metrics.json",
                "best_predictions.csv",
                "best_debug_predictions.jsonl",
                "modality_cache.jsonl",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)

            best_config = json.loads((output_dir / "best_fusion_config.json").read_text(encoding="utf-8"))
            self.assertEqual(best_config["metric"], "macro_f1")
            self.assertEqual(best_config["labels"], CANONICAL_LABELS)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
