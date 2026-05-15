from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS, label_to_id, normalize_label
from multimodal_emotion.inference.fusion import FUSION_MODES, FusionEngine
from multimodal_emotion.inference.result import PredictionResult
from multimodal_emotion.inference.runtime_config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EmoVision inference on a JSONL or CSV manifest.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL or CSV manifest.")
    parser.add_argument("--modality", required=True, choices=["text", "audio", "video", "fusion"])
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs.")
    parser.add_argument("--fusion-mode", default="weighted_probs", choices=sorted(FUSION_MODES))
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum sample count.")
    parser.add_argument("--device", default="auto", help="Device for model inference, e.g. auto, cpu, cuda.")
    return parser.parse_args()


def read_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file was not found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on manifest line {line_no}: {error}") from error

    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return rows


def _field(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _true_label(row: dict[str, Any]) -> str:
    raw = _field(row, "label", "true_label")
    if raw is None:
        raise ValueError(f"Manifest row is missing label/true_label: {row}")
    label = normalize_label(raw)
    if label not in label_to_id:
        raise ValueError(f"Unknown true label {raw!r} after normalization.")
    return label


def unavailable(modality: str, error: str) -> PredictionResult:
    return PredictionResult.from_unavailable(modality, error)


def fusion_from_single(modality: str, result: PredictionResult, mode: str) -> PredictionResult:
    if not result.available or result.probs is None:
        return PredictionResult.from_unavailable("fusion", result.error or f"{modality} unavailable.")
    pred_idx = int(np.argmax(np.asarray(result.probs, dtype=np.float64)))
    return PredictionResult(
        modality="fusion",
        available=True,
        labels=list(CANONICAL_LABELS),
        logits=result.logits,
        probs=[float(value) for value in result.probs],
        pred_label=CANONICAL_LABELS[pred_idx],
        confidence=float(result.probs[pred_idx]),
        quality={"mode": mode, "weights_used": {modality: 1.0}, "single_modality": modality},
    )


def build_predictors(modality: str, device: str = "auto") -> dict[str, Any]:
    predictors: dict[str, Any] = {}
    runtime_config = load_runtime_config(validate_paths=True)
    if modality in {"text", "fusion"}:
        from multimodal_emotion.inference.text import TextEmotionPredictor

        predictors["text"] = TextEmotionPredictor(
            model_path=runtime_config.model_paths.text,
            temperature=runtime_config.temperatures["text"],
            device=device,
        )
    if modality in {"audio", "fusion"}:
        from multimodal_emotion.inference.audio import AudioEmotionPredictor

        predictors["audio"] = AudioEmotionPredictor(
            model_path=runtime_config.model_paths.audio,
            temperature=runtime_config.temperatures["audio"],
            device=device,
        )
    if modality in {"video", "fusion"}:
        from multimodal_emotion.inference.video import VideoEmotionPredictor

        predictors["video"] = VideoEmotionPredictor(
            model_path=runtime_config.model_paths.video,
            temperature=runtime_config.temperatures["video"],
            device=device,
        )
    return predictors


def predict_row(
    row: dict[str, Any],
    *,
    modality: str,
    predictors: dict[str, Any],
    fusion_engine: FusionEngine,
) -> dict[str, PredictionResult]:
    text_value = _field(row, "text")
    audio_path = _field(row, "audio_path")
    video_path = _field(row, "video_path")

    text_result = unavailable("text", "Text field is missing.")
    audio_result = unavailable("audio", "audio_path field is missing.")
    video_result = unavailable("video", "video_path field is missing.")

    if modality in {"text", "fusion"} and "text" in predictors:
        text_result = predictors["text"].predict(text_value)
    if modality in {"audio", "fusion"} and "audio" in predictors:
        audio_result = predictors["audio"].predict(audio_path) if audio_path else audio_result
    if modality in {"video", "fusion"} and "video" in predictors:
        video_result = predictors["video"].predict(video_path) if video_path else video_result

    if modality == "fusion":
        fusion_result = fusion_engine.fuse([text_result, audio_result, video_result])
    elif modality == "text":
        fusion_result = fusion_from_single("text", text_result, fusion_engine.mode)
    elif modality == "audio":
        fusion_result = fusion_from_single("audio", audio_result, fusion_engine.mode)
    elif modality == "video":
        fusion_result = fusion_from_single("video", video_result, fusion_engine.mode)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return {
        "text": text_result,
        "audio": audio_result,
        "video": video_result,
        "fusion": fusion_result,
    }


def _prediction_csv_row(sample_id: str, true_label: str, result: PredictionResult) -> dict[str, Any]:
    probs = result.probs or [0.0] * len(CANONICAL_LABELS)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "true_label": true_label,
        "pred_label": result.pred_label or "",
        "confidence": float(result.confidence),
    }
    for index, label in enumerate(CANONICAL_LABELS):
        row[f"prob_{label}"] = float(probs[index])
    return row


def collapse_warnings_from_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    threshold: float = 0.80,
) -> list[str]:
    total = len(prediction_rows)
    if total <= 0:
        return []

    counts: dict[str, int] = {}
    for row in prediction_rows:
        pred_label = row.get("pred_label")
        if pred_label:
            counts[str(pred_label)] = counts.get(str(pred_label), 0) + 1
    if not counts:
        return []

    label, count = max(counts.items(), key=lambda item: item[1])
    if (count / float(total)) > threshold:
        return [f"WARNING: predictions collapsed: {label} = {count}/{total}"]
    return []


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    modality: str,
    output_dir: str | Path,
    fusion_mode: str = "weighted_probs",
    predictors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Manifest contains no rows to evaluate.")

    predictors = predictors or {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runtime_config = load_runtime_config(validate_paths=False)
    fusion_engine = FusionEngine(
        global_weights=runtime_config.global_weights,
        mode=fusion_mode,
        confidence_gating=runtime_config.confidence_gating,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        sample_id = _field(row, "sample_id") or str(index)
        true_label = _true_label(row)
        results = predict_row(row, modality=modality, predictors=predictors, fusion_engine=fusion_engine)
        selected = results["fusion"]
        if not selected.available or selected.pred_label is None or selected.probs is None:
            selected = PredictionResult(
                modality="fusion",
                available=True,
                labels=list(CANONICAL_LABELS),
                logits=None,
                probs=[1.0 / len(CANONICAL_LABELS)] * len(CANONICAL_LABELS),
                pred_label=CANONICAL_LABELS[0],
                confidence=1.0 / len(CANONICAL_LABELS),
                quality={
                    "mode": fusion_mode,
                    "weights_used": {},
                    "fallback": "uniform_after_unavailable_prediction",
                },
                error=results["fusion"].error,
            )

        y_true.append(label_to_id[true_label])
        y_pred.append(label_to_id[selected.pred_label])
        prediction_rows.append(_prediction_csv_row(sample_id, true_label, selected))
        debug_rows.append(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "text": results["text"].to_dict(),
                "audio": results["audio"].to_dict(),
                "video": results["video"].to_dict(),
                "fusion": selected.to_dict(),
                "fusion_debug": {
                    "mode": selected.quality.get("mode", fusion_mode),
                    "weights_used": selected.quality.get("weights_used", {}),
                },
            }
        )

    collapse_warnings = collapse_warnings_from_prediction_rows(prediction_rows)
    metrics = {
        "primary_metric": "macro_f1",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "num_samples": len(rows),
        "modality": modality,
        "fusion_mode": fusion_mode,
        "labels": list(CANONICAL_LABELS),
        "collapse_warnings": list(collapse_warnings),
    }

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CANONICAL_LABELS))),
        target_names=CANONICAL_LABELS,
        zero_division=0,
    )
    (output_path / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CANONICAL_LABELS))))
    with (output_path / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *CANONICAL_LABELS])
        for row_label, values in zip(CANONICAL_LABELS, matrix, strict=True):
            writer.writerow([row_label, *[int(value) for value in values]])

    prediction_columns = [
        "sample_id",
        "true_label",
        "pred_label",
        "confidence",
        *[f"prob_{label}" for label in CANONICAL_LABELS],
    ]
    with (output_path / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    with (output_path / "debug_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for debug_row in debug_rows:
            handle.write(json.dumps(debug_row, ensure_ascii=True) + "\n")

    for warning in collapse_warnings:
        print(warning)

    return metrics


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest, args.limit)
    predictors = build_predictors(args.modality, device=args.device)
    metrics = evaluate_rows(
        rows,
        modality=args.modality,
        output_dir=args.output_dir,
        fusion_mode=args.fusion_mode,
        predictors=predictors,
    )
    print(f"Wrote evaluation outputs to {args.output_dir}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
