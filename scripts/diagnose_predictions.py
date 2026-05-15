from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose prediction collapse and modality quality.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv.")
    parser.add_argument("--debug-jsonl", help="Optional debug_predictions.jsonl path.")
    parser.add_argument("--output-dir", help="Optional directory for diagnostics.json and diagnostics.txt.")
    return parser.parse_args()


def read_predictions(path: str | Path) -> list[dict[str, str]]:
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Predictions file was not found: {prediction_path}")
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_debug_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    debug_path = Path(path)
    if not debug_path.is_file():
        raise FileNotFoundError(f"Debug JSONL file was not found: {debug_path}")
    rows: list[dict[str, Any]] = []
    with debug_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on debug JSONL line {line_no}: {error}") from error
    return rows


def distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def confidence_by_label(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get("pred_label", ""))
        if not label:
            continue
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(label, []).append(confidence)
    return {label: float(mean(values)) for label, values in sorted(grouped.items())}


def collapse_warnings_from_prediction_rows(
    prediction_rows: list[dict[str, str]],
    *,
    threshold: float = 0.80,
) -> list[str]:
    labels = [str(row.get("pred_label", "")) for row in prediction_rows if row.get("pred_label")]
    total = len(prediction_rows)
    if total <= 0 or not labels:
        return []
    counts = distribution(labels)
    label, count = max(counts.items(), key=lambda item: item[1])
    if (count / float(total)) > threshold:
        return [f"WARNING: predictions collapsed: {label} = {count}/{total}"]
    return []


def label_metrics(rows: list[dict[str, str]]) -> dict[str, float | None]:
    pairs = [
        (str(row.get("true_label") or row.get("label") or ""), str(row.get("pred_label", "")))
        for row in rows
        if (row.get("true_label") or row.get("label")) and row.get("pred_label")
    ]
    if not pairs:
        return {"accuracy": None, "macro_f1": None}
    y_true = [item[0] for item in pairs]
    y_pred = [item[1] for item in pairs]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def quality_values(debug_rows: list[dict[str, Any]], modality: str, key: str, *aliases: str) -> list[float]:
    values: list[float] = []
    keys = (key, *aliases)
    for row in debug_rows:
        result = row.get(modality, {})
        quality = result.get("quality", {}) if isinstance(result, dict) else {}
        if not isinstance(quality, dict):
            continue
        raw_value = None
        for candidate_key in keys:
            if quality.get(candidate_key) is not None:
                raw_value = quality[candidate_key]
                break
        if raw_value is None:
            continue
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            continue
    return values


def modality_unavailable_count(debug_rows: list[dict[str, Any]], modality: str) -> int:
    count = 0
    for row in debug_rows:
        result = row.get(modality, {})
        if isinstance(result, dict) and not bool(result.get("available", False)):
            count += 1
    return count


def build_diagnostics(prediction_rows: list[dict[str, str]], debug_rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidences: list[float] = []
    for row in prediction_rows:
        try:
            confidences.append(float(row.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue

    metrics = label_metrics(prediction_rows)
    per_label_confidence = confidence_by_label(prediction_rows)
    diagnostics: dict[str, Any] = {
        "num_predictions": len(prediction_rows),
        "pred_label_distribution": distribution([str(row.get("pred_label", "")) for row in prediction_rows]),
        "true_label_distribution": distribution(
            [
                str(row.get("true_label") or row.get("label") or "")
                for row in prediction_rows
                if row.get("true_label") or row.get("label")
            ]
        ),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "average_confidence": float(mean(confidences)) if confidences else None,
        "median_confidence": float(median(confidences)) if confidences else None,
        "per_pred_label_confidence_mean": per_label_confidence,
        "per_label_confidence_mean": per_label_confidence,
        "missing_pred_label_count": sum(1 for row in prediction_rows if not row.get("pred_label")),
        "collapse_warnings": collapse_warnings_from_prediction_rows(prediction_rows),
    }

    if debug_rows:
        diagnostics["unavailable_count"] = {
            modality: modality_unavailable_count(debug_rows, modality)
            for modality in ("text", "audio", "video", "fusion")
        }
        diagnostics["audio_quality"] = {
            "duration_sec": numeric_summary(quality_values(debug_rows, "audio", "duration_sec")),
            "rms": numeric_summary(quality_values(debug_rows, "audio", "rms")),
            "num_chunks": numeric_summary(quality_values(debug_rows, "audio", "num_chunks")),
            "quality_weight_multiplier": numeric_summary(
                quality_values(debug_rows, "audio", "quality_weight_multiplier")
            ),
        }
        diagnostics["video_quality"] = {
            "face_ratio": numeric_summary(quality_values(debug_rows, "video", "face_ratio")),
            "face_frames": numeric_summary(
                quality_values(debug_rows, "video", "face_frames", "face_detected_frames")
            ),
            "sampled_frames": numeric_summary(quality_values(debug_rows, "video", "sampled_frames")),
            "quality_weight_multiplier": numeric_summary(
                quality_values(debug_rows, "video", "quality_weight_multiplier")
            ),
        }
    return diagnostics


def format_diagnostics(diagnostics: dict[str, Any]) -> str:
    lines = [
        f"num_predictions: {diagnostics['num_predictions']}",
        f"pred_label_distribution: {diagnostics['pred_label_distribution']}",
        f"true_label_distribution: {diagnostics['true_label_distribution']}",
        f"accuracy: {diagnostics['accuracy']}",
        f"macro_f1: {diagnostics['macro_f1']}",
        f"average_confidence: {diagnostics['average_confidence']}",
        f"median_confidence: {diagnostics['median_confidence']}",
        f"per_pred_label_confidence_mean: {diagnostics['per_pred_label_confidence_mean']}",
        f"missing_pred_label_count: {diagnostics['missing_pred_label_count']}",
    ]
    for warning in diagnostics.get("collapse_warnings", []):
        lines.append(str(warning))
    if "unavailable_count" in diagnostics:
        lines.append(f"unavailable_count: {diagnostics['unavailable_count']}")
        lines.append(f"audio_quality: {diagnostics['audio_quality']}")
        lines.append(f"video_quality: {diagnostics['video_quality']}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    prediction_rows = read_predictions(args.predictions)
    debug_rows = read_debug_jsonl(args.debug_jsonl)
    diagnostics = build_diagnostics(prediction_rows, debug_rows)
    text = format_diagnostics(diagnostics)
    print(text)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        (output_dir / "diagnostics.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
