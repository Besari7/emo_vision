from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_emotion.labels import CANONICAL_LABELS, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics from saved prediction and label arrays.")
    parser.add_argument("--preds", required=True, help="Path to .npy predictions. Supports class ids, labels, or 2D probabilities.")
    parser.add_argument("--labels", required=True, help="Path to .npy true labels. Supports class ids or labels.")
    parser.add_argument("--output-dir", required=True, help="Directory for test_metrics.json, classification_report.csv, and confusion_matrix.png.")
    parser.add_argument("--label-names", nargs="*", default=None, help="Label names for integer class ids, in model output index order.")
    parser.add_argument("--model-config", help="Hugging Face config.json containing id2label for integer/probability predictions.")
    return parser.parse_args()


def load_npy(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() != ".npy":
        raise ValueError(f"Only .npy inputs are supported: {source}")
    return np.load(source, allow_pickle=False)


def labels_from_model_config(path: str | None) -> list[str] | None:
    if not path:
        return None

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        raise ValueError(f"Model config does not contain an id2label mapping: {config_path}")

    indexed_labels: list[tuple[int, str]] = []
    for raw_index, raw_label in id2label.items():
        if not isinstance(raw_label, str):
            raise ValueError(f"Model config id2label value is not a string: {raw_index} -> {raw_label!r}")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Model config id2label key is not an integer index: {raw_index!r}") from error
        indexed_labels.append((index, raw_label))

    if not indexed_labels:
        raise ValueError(f"Model config id2label mapping is empty: {config_path}")
    return [normalize_label(label) for _, label in sorted(indexed_labels)]


def requires_index_labels(values: np.ndarray, *, is_prediction: bool) -> bool:
    if is_prediction and values.ndim == 2:
        return True
    if values.ndim == 1 and np.issubdtype(values.dtype, np.integer):
        return True
    return False


def to_label_array(values: np.ndarray, label_names: list[str], *, is_prediction: bool) -> np.ndarray:
    if is_prediction and values.ndim == 2:
        values = np.argmax(values, axis=1)
    elif values.ndim != 1:
        raise ValueError(f"Expected a 1D array or 2D prediction probabilities, received shape {values.shape}.")

    if np.issubdtype(values.dtype, np.integer):
        labels: list[str] = []
        for value in values:
            index = int(value)
            if index < 0 or index >= len(label_names):
                raise ValueError(f"Class index {index} is outside label_names range.")
            labels.append(normalize_label(label_names[index]))
        return np.asarray(labels)

    return np.asarray([normalize_label(str(value)) for value in values])


def draw_confusion_matrix_png(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    cell = 86
    margin_left = 150
    margin_top = 120
    width = margin_left + cell * len(labels) + 40
    height = margin_top + cell * len(labels) + 40
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    max_value = int(matrix.max()) if matrix.size else 0

    for idx, label in enumerate(labels):
        x = margin_left + idx * cell + 8
        y = margin_top - 34
        draw.text((x, y), label[:10], fill="black")
        draw.text((16, margin_top + idx * cell + 32), label[:16], fill="black")

    draw.text((margin_left, 18), "Predicted label", fill="black")
    draw.text((16, margin_top - 70), "True label", fill="black")

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = int(matrix[row, col])
            intensity = 255 if max_value == 0 else int(255 - (180 * value / max_value))
            fill = (intensity, intensity, 255)
            x1 = margin_left + col * cell
            y1 = margin_top + row * cell
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rectangle((x1, y1, x2, y2), fill=fill, outline="black")
            draw.text((x1 + 32, y1 + 32), str(value), fill="black")

    image.save(output_path)


def main() -> None:
    args = parse_args()
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_preds = load_npy(args.preds)
    raw_labels = load_npy(args.labels)

    label_source = "canonical default"
    if args.label_names is not None:
        if not args.label_names:
            raise ValueError("--label-names was provided without any labels.")
        label_names = [normalize_label(label) for label in args.label_names]
        label_source = "--label-names"
    else:
        config_labels = labels_from_model_config(args.model_config)
        if config_labels is not None:
            label_names = config_labels
            label_source = "--model-config id2label"
        elif requires_index_labels(raw_preds, is_prediction=True) or requires_index_labels(raw_labels, is_prediction=False):
            raise ValueError(
                "Integer class ids or 2D probability predictions require an explicit label order. "
                "Pass --model-config path\\to\\config.json or --label-names in model output index order."
            )
        else:
            label_names = [normalize_label(label) for label in CANONICAL_LABELS]

    preds = to_label_array(raw_preds, label_names, is_prediction=True)
    labels = to_label_array(raw_labels, label_names, is_prediction=False)

    if preds.shape[0] != labels.shape[0]:
        raise ValueError(f"Prediction count {preds.shape[0]} does not match label count {labels.shape[0]}.")

    observed = set(preds.tolist()) | set(labels.tolist())
    unknown = sorted(label for label in observed if label not in label_names)
    if unknown:
        raise ValueError(f"Observed labels are missing from label_names: {unknown}")

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=label_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, labels=label_names, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, labels=label_names, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, labels=label_names, average="macro", zero_division=0)),
        "num_samples": int(labels.shape[0]),
        "labels": label_names,
        "label_source": label_source,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(labels, preds, labels=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    matrix = confusion_matrix(labels, preds, labels=label_names)
    draw_confusion_matrix_png(matrix, label_names, output_dir / "confusion_matrix.png")
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()
