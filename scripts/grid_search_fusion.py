from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS, label_to_id
from multimodal_emotion.inference.fusion import FusionEngine
from multimodal_emotion.inference.result import PredictionResult
from scripts.evaluate_inference import _field, _prediction_csv_row, _true_label, build_predictors, read_manifest


TEXT_TEMPERATURES = [0.8, 1.0, 1.2]
AUDIO_TEMPERATURES = [1.0, 1.4, 2.0, 3.0, 4.0, 5.0]
VIDEO_TEMPERATURES = [1.0, 1.3, 1.6, 2.0]

TEXT_WEIGHTS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
AUDIO_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
VIDEO_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search fusion temperatures and global weights on a validation manifest. "
            "Do not tune on a held-out test set."
        )
    )
    parser.add_argument("--manifest", required=True, help="Validation manifest path. Do not pass a test manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for grid-search outputs.")
    parser.add_argument("--fusion-mode", default="weighted_probs", choices=["weighted_probs", "log_probs", "both"])
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum sample count.")
    parser.add_argument("--device", default="auto", help="Device for model inference when cache is not used.")
    parser.add_argument("--cache-dir", default=None, help="Optional directory containing/writing modality_cache.jsonl.")
    parser.add_argument("--metric", default="macro_f1", choices=["macro_f1"], help="Selection metric.")
    parser.add_argument("--use-cache", action="store_true", help="Read modality_cache.jsonl instead of running models.")
    return parser.parse_args()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0:
        return np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
    return exp_values / total


def prediction_from_dict(payload: dict[str, Any], modality: str) -> PredictionResult:
    if not payload:
        return PredictionResult.from_unavailable(modality, "Missing cached prediction.")
    return PredictionResult(
        modality=modality,  # type: ignore[arg-type]
        available=bool(payload.get("available", False)),
        labels=list(payload.get("labels") or CANONICAL_LABELS),
        logits=payload.get("logits"),
        probs=payload.get("probs"),
        pred_label=payload.get("pred_label"),
        confidence=float(payload.get("confidence", 0.0)),
        quality=dict(payload.get("quality") or {}),
        error=payload.get("error"),
    )


def temperature_scaled_prediction(prediction: PredictionResult, temperature: float) -> PredictionResult:
    if not prediction.available:
        return prediction

    if prediction.logits is not None:
        logits = np.asarray(prediction.logits, dtype=np.float64)
        scaled_probs = softmax(logits / max(float(temperature), 1e-6))
    elif prediction.probs is not None:
        base_probs = np.asarray(prediction.probs, dtype=np.float64)
        base_probs = np.clip(base_probs, 1e-12, None)
        scaled_probs = softmax(np.log(base_probs) / max(float(temperature), 1e-6))
    else:
        return PredictionResult.from_unavailable(
            prediction.modality,
            prediction.error or "Cached prediction has neither logits nor probabilities.",
            prediction.quality,
        )

    pred_idx = int(np.argmax(scaled_probs))
    quality = dict(prediction.quality)
    quality["grid_temperature"] = float(temperature)
    return PredictionResult(
        modality=prediction.modality,
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=prediction.logits,
        probs=[float(value) for value in scaled_probs],
        pred_label=CANONICAL_LABELS[pred_idx],
        confidence=float(scaled_probs[pred_idx]),
        quality=quality,
        error=prediction.error,
    )


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {name: max(float(value), 0.0) for name, value in weights.items()}
    total = float(sum(clipped.values()))
    if total <= 0.0:
        raise ValueError("At least one fusion weight must be positive.")
    return {name: value / total for name, value in clipped.items()}


def candidate_modes(fusion_mode: str) -> list[str]:
    if fusion_mode == "both":
        return ["weighted_probs", "log_probs"]
    return [fusion_mode]


def iter_grid(fusion_mode: str):
    for mode in candidate_modes(fusion_mode):
        for text_temp, audio_temp, video_temp in itertools.product(
            TEXT_TEMPERATURES,
            AUDIO_TEMPERATURES,
            VIDEO_TEMPERATURES,
        ):
            temperatures = {"text": text_temp, "audio": audio_temp, "video": video_temp}
            for text_weight, audio_weight, video_weight in itertools.product(
                TEXT_WEIGHTS,
                AUDIO_WEIGHTS,
                VIDEO_WEIGHTS,
            ):
                raw_weights = {"text": text_weight, "audio": audio_weight, "video": video_weight}
                if sum(raw_weights.values()) <= 0.0:
                    continue
                yield {
                    "fusion_mode": mode,
                    "temperatures": temperatures,
                    "weights": normalize_weights(raw_weights),
                }


def unavailable(modality: str, error: str) -> PredictionResult:
    return PredictionResult.from_unavailable(modality, error)


def safe_predict(predictor: Any, modality: str, value: str | None) -> PredictionResult:
    if not value:
        return unavailable(modality, f"{modality} input is missing.")
    try:
        return predictor.predict(value)
    except Exception as error:
        return unavailable(modality, f"{modality} prediction failed: {error}")


def build_modality_cache(
    rows: list[dict[str, Any]],
    *,
    device: str,
    cache_path: Path,
) -> list[dict[str, Any]]:
    predictors = build_predictors("fusion", device=device)
    cache_rows: list[dict[str, Any]] = []

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            sample_id = _field(row, "sample_id") or str(index)
            true_label = _true_label(row)
            text_value = _field(row, "text")
            audio_path = _field(row, "audio_path")
            video_path = _field(row, "video_path")

            text_result = safe_predict(predictors["text"], "text", text_value)
            audio_result = safe_predict(predictors["audio"], "audio", audio_path)
            video_result = safe_predict(predictors["video"], "video", video_path)

            cache_row = {
                "sample_id": sample_id,
                "true_label": true_label,
                "text": text_result.to_dict(),
                "audio": audio_result.to_dict(),
                "video": video_result.to_dict(),
            }
            cache_rows.append(cache_row)
            handle.write(json.dumps(cache_row, ensure_ascii=True) + "\n")
    return cache_rows


def read_modality_cache(cache_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modality cache was not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on cache line {line_no}: {error}") from error
    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return rows


def copy_cache_to_output(cache_path: Path, output_cache_path: Path) -> None:
    if cache_path.resolve() == output_cache_path.resolve():
        return
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache_path, output_cache_path)


def row_predictions_for_config(cache_row: dict[str, Any], config: dict[str, Any]) -> list[PredictionResult]:
    temperatures = config["temperatures"]
    return [
        temperature_scaled_prediction(prediction_from_dict(cache_row.get("text", {}), "text"), temperatures["text"]),
        temperature_scaled_prediction(prediction_from_dict(cache_row.get("audio", {}), "audio"), temperatures["audio"]),
        temperature_scaled_prediction(prediction_from_dict(cache_row.get("video", {}), "video"), temperatures["video"]),
    ]


def fallback_fusion_result(error: str, fusion_mode: str) -> PredictionResult:
    probs = [1.0 / len(CANONICAL_LABELS)] * len(CANONICAL_LABELS)
    return PredictionResult(
        modality="fusion",
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=None,
        probs=probs,
        pred_label=CANONICAL_LABELS[0],
        confidence=probs[0],
        quality={"mode": fusion_mode, "weights_used": {}, "fallback": "uniform_after_unavailable_prediction"},
        error=error,
    )


def fuse_cache_row(cache_row: dict[str, Any], config: dict[str, Any]) -> tuple[PredictionResult, list[PredictionResult]]:
    predictions = row_predictions_for_config(cache_row, config)
    engine = FusionEngine(global_weights=config["weights"], mode=config["fusion_mode"])
    fused = engine.fuse(predictions)
    if not fused.available or fused.pred_label is None or fused.probs is None:
        fused = fallback_fusion_result(fused.error or "No available modalities for fusion.", config["fusion_mode"])
    return fused, predictions


def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    true_values = np.asarray(y_true, dtype=np.int64)
    pred_values = np.asarray(y_pred, dtype=np.int64)
    if true_values.size == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
        }

    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    supports: list[int] = []
    macro_indices: list[int] = []
    for class_index in range(len(CANONICAL_LABELS)):
        true_mask = true_values == class_index
        pred_mask = pred_values == class_index
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        fp = int(np.logical_and(~true_mask, pred_mask).sum())
        fn = int(np.logical_and(true_mask, ~pred_mask).sum())
        support = int(true_mask.sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        supports.append(support)
        if support > 0 or int(pred_mask.sum()) > 0:
            macro_indices.append(class_index)

    support_values = np.asarray(supports, dtype=np.float64)
    support_total = float(support_values.sum())
    weights = support_values / support_total if support_total > 0.0 else np.zeros_like(support_values)
    if not macro_indices:
        macro_indices = list(range(len(CANONICAL_LABELS)))
    return {
        "accuracy": float((true_values == pred_values).mean()),
        "macro_precision": float(np.mean([precisions[index] for index in macro_indices])),
        "macro_recall": float(np.mean([recalls[index] for index in macro_indices])),
        "macro_f1": float(np.mean([f1_values[index] for index in macro_indices])),
        "weighted_precision": float(np.dot(weights, np.asarray(precisions, dtype=np.float64))),
        "weighted_recall": float(np.dot(weights, np.asarray(recalls, dtype=np.float64))),
        "weighted_f1": float(np.dot(weights, np.asarray(f1_values, dtype=np.float64))),
    }


def evaluate_config(cache_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for cache_row in cache_rows:
        true_label = str(cache_row["true_label"])
        fused, _ = fuse_cache_row(cache_row, config)
        y_true.append(label_to_id[true_label])
        y_pred.append(label_to_id[fused.pred_label or CANONICAL_LABELS[0]])

    metrics = metrics_from_predictions(y_true, y_pred)
    return {
        **config,
        **metrics,
        "num_samples": len(cache_rows),
    }


def result_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "fusion_mode": result["fusion_mode"],
        "temperature_text": result["temperatures"]["text"],
        "temperature_audio": result["temperatures"]["audio"],
        "temperature_video": result["temperatures"]["video"],
        "weight_text": result["weights"]["text"],
        "weight_audio": result["weights"]["audio"],
        "weight_video": result["weights"]["video"],
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "weighted_precision": result["weighted_precision"],
        "weighted_recall": result["weighted_recall"],
        "weighted_f1": result["weighted_f1"],
        "num_samples": result["num_samples"],
    }


def write_grid_results(results: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "fusion_mode",
        "temperature_text",
        "temperature_audio",
        "temperature_video",
        "weight_text",
        "weight_audio",
        "weight_video",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "num_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result_csv_row(result))


def write_best_outputs(cache_rows: list[dict[str, Any]], best: dict[str, Any], output_dir: Path, metric: str) -> None:
    prediction_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for cache_row in cache_rows:
        sample_id = str(cache_row["sample_id"])
        true_label = str(cache_row["true_label"])
        fused, modality_predictions = fuse_cache_row(cache_row, best)

        y_true.append(label_to_id[true_label])
        y_pred.append(label_to_id[fused.pred_label or CANONICAL_LABELS[0]])
        prediction_rows.append(_prediction_csv_row(sample_id, true_label, fused))
        debug_rows.append(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "text": modality_predictions[0].to_dict(),
                "audio": modality_predictions[1].to_dict(),
                "video": modality_predictions[2].to_dict(),
                "fusion": fused.to_dict(),
                "fusion_debug": {
                    "mode": fused.quality.get("mode", best["fusion_mode"]),
                    "weights_used": fused.quality.get("weights_used", {}),
                },
            }
        )

    best_metrics = {
        "primary_metric": metric,
        **metrics_from_predictions(y_true, y_pred),
        "num_samples": len(cache_rows),
        "labels": list(CANONICAL_LABELS),
        "fusion_mode": best["fusion_mode"],
        "temperatures": best["temperatures"],
        "weights": best["weights"],
    }
    (output_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

    best_config = {
        "fusion_mode": best["fusion_mode"],
        "temperatures": best["temperatures"],
        "weights": best["weights"],
        "metric": metric,
        "metric_value": float(best[metric]),
        "labels": list(CANONICAL_LABELS),
        "note": "Validation/domain-specific recommendation only. Do not apply automatically or tune on test data.",
    }
    (output_dir / "best_fusion_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    prediction_columns = [
        "sample_id",
        "true_label",
        "pred_label",
        "confidence",
        *[f"prob_{label}" for label in CANONICAL_LABELS],
    ]
    with (output_dir / "best_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (output_dir / "best_debug_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for debug_row in debug_rows:
            handle.write(json.dumps(debug_row, ensure_ascii=True) + "\n")


def grid_search(cache_rows: list[dict[str, Any]], *, fusion_mode: str, metric: str) -> list[dict[str, Any]]:
    if not cache_rows:
        raise ValueError("No cached predictions available for grid search.")
    results: list[dict[str, Any]] = []
    for config in iter_grid(fusion_mode):
        results.append(evaluate_config(cache_rows, config))
    return sorted(results, key=lambda item: float(item[metric]), reverse=True)


def print_top_configs(results: list[dict[str, Any]], metric: str, top_n: int = 10) -> None:
    print(f"Top {min(top_n, len(results))} fusion configs by {metric}:")
    for rank, result in enumerate(results[:top_n], start=1):
        temps = result["temperatures"]
        weights = result["weights"]
        print(
            f"{rank:02d}. {metric}={result[metric]:.6f} mode={result['fusion_mode']} "
            f"T(text/audio/video)={temps['text']}/{temps['audio']}/{temps['video']} "
            f"W(text/audio/video)={weights['text']:.4f}/{weights['audio']:.4f}/{weights['video']:.4f}"
        )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir
    cache_path = cache_dir / "modality_cache.jsonl"
    output_cache_path = output_dir / "modality_cache.jsonl"

    print("WARNING: Run grid search on validation data only. Do not tune fusion on a held-out test set.")
    rows = read_manifest(args.manifest, args.limit)

    if args.use_cache:
        cache_rows = read_modality_cache(cache_path, args.limit)
    else:
        cache_rows = build_modality_cache(rows, device=args.device, cache_path=cache_path)
    copy_cache_to_output(cache_path, output_cache_path)

    results = grid_search(cache_rows, fusion_mode=args.fusion_mode, metric=args.metric)
    write_grid_results(results, output_dir / "grid_search_results.csv")
    best = results[0]
    write_best_outputs(cache_rows, best, output_dir, args.metric)
    print_top_configs(results, args.metric)
    print(f"Wrote grid-search outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
