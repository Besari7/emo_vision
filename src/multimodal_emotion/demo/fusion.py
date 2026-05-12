from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multimodal_emotion.labels import CANONICAL_LABELS, normalize_label, reorder_probabilities


COMMON_LABELS = list(CANONICAL_LABELS)

MODALITY_BASE_WEIGHTS = {
    "text": 0.40,
    "audio": 0.35,
    "video": 0.25,
}

TEXT_LABEL_MAP: dict[str, dict[str, float]] = {
    "admiration": {"joy": 1.0},
    "amusement": {"joy": 1.0},
    "anger": {"anger": 1.0},
    "annoyance": {"anger": 1.0},
    "approval": {"joy": 1.0},
    "caring": {"joy": 1.0},
    "confusion": {"surprise": 1.0},
    "curiosity": {"surprise": 1.0},
    "desire": {"joy": 1.0},
    "disappointment": {"sadness": 1.0},
    "disapproval": {"disgust": 0.6, "anger": 0.4},
    "disgust": {"disgust": 1.0},
    "embarrassment": {"sadness": 1.0},
    "excitement": {"joy": 0.8, "surprise": 0.2},
    "fear": {"fear": 1.0},
    "gratitude": {"joy": 1.0},
    "grief": {"sadness": 1.0},
    "joy": {"joy": 1.0},
    "love": {"joy": 0.85, "surprise": 0.15},
    "nervousness": {"fear": 1.0},
    "neutral": {"neutral": 1.0},
    "optimism": {"joy": 1.0},
    "pride": {"joy": 1.0},
    "realization": {"surprise": 1.0},
    "relief": {"joy": 1.0},
    "remorse": {"sadness": 1.0},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
}

AUDIO_LABEL_MAP: dict[str, dict[str, float]] = {
    "anger": {"anger": 1.0},
    "calm": {"neutral": 1.0},
    "disgust": {"disgust": 1.0},
    "fear": {"fear": 1.0},
    "fearful": {"fear": 1.0},
    "happy": {"joy": 1.0},
    "joy": {"joy": 1.0},
    "neutral": {"neutral": 1.0},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
    "surprised": {"surprise": 1.0},
}

VIDEO_LABEL_MAP: dict[str, dict[str, float]] = {
    "angry": {"anger": 1.0},
    "anger": {"anger": 1.0},
    "disgust": {"disgust": 1.0},
    "disgusted": {"disgust": 1.0},
    "fear": {"fear": 1.0},
    "fearful": {"fear": 1.0},
    "happy": {"joy": 1.0},
    "joy": {"joy": 1.0},
    "neutral": {"neutral": 1.0},
    "sad": {"sadness": 1.0},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
    "surprised": {"surprise": 1.0},
}

@dataclass(slots=True)
class ModalitySummary:
    name: str
    status: str
    probabilities: dict[str, float]
    confidence: float
    quality: float
    note: str


def remap_predictions(
    predictions: list[dict],
    label_map: dict[str, dict[str, float]],
    *,
    smoothing: float = 1e-6,
) -> dict[str, float]:
    scores = np.full(len(COMMON_LABELS), smoothing, dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(COMMON_LABELS)}

    for prediction in predictions:
        raw_label = normalize_label(str(prediction["label"]))
        if raw_label not in label_map:
            raise ValueError(f"Prediction label {prediction['label']!r} is not mapped to a canonical label.")
        for target_label, weight in label_map[raw_label].items():
            scores[label_to_index[target_label]] += float(prediction["score"]) * weight

    scores = scores / scores.sum()
    return {label: float(scores[index]) for index, label in enumerate(COMMON_LABELS)}


def confidence_from_scores(probabilities: dict[str, float]) -> float:
    return float(max(probabilities.values(), default=0.0))


def weighted_fusion(modalities: list[ModalitySummary]) -> dict:
    fused = np.zeros(len(COMMON_LABELS), dtype=np.float64)
    modality_weights: dict[str, float] = {}

    for modality in modalities:
        if modality.status not in {"ok", "fallback"}:
            continue

        base_weight = MODALITY_BASE_WEIGHTS[modality.name]
        quality_weight = max(modality.quality, 0.20)
        confidence_weight = max(modality.confidence, 0.10)
        final_weight = base_weight * quality_weight * confidence_weight
        modality_weights[modality.name] = float(final_weight)

        # Canonical order is required here; otherwise vector indices can mix
        # emotions across modalities during probability-level fusion.
        aligned_probabilities = reorder_probabilities(
            modality.probabilities,
            source_labels=list(modality.probabilities.keys()),
            target_labels=COMMON_LABELS,
        )
        fused += final_weight * np.array([aligned_probabilities[label] for label in COMMON_LABELS])

    if not modality_weights:
        raise ValueError("At least one modality must be available for fusion.")

    fused = fused / fused.sum()
    probabilities = {label: float(fused[index]) for index, label in enumerate(COMMON_LABELS)}
    predicted_label = max(probabilities, key=probabilities.get)
    return {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "confidence": float(probabilities[predicted_label]),
        "modality_weights": modality_weights,
    }
