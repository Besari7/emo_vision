from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CANONICAL_LABELS = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
]

LABEL_ALIASES = {
    "ang": "anger",
    "anger": "anger",
    "angry": "anger",
    "calm": "neutral",
    "dis": "disgust",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fea": "fear",
    "fear": "fear",
    "fearful": "fear",
    "fearfulness": "fear",
    "hap": "joy",
    "happiness": "joy",
    "happy": "joy",
    "joy": "joy",
    "neu": "neutral",
    "neutral": "neutral",
    "sad": "sadness",
    "sadness": "sadness",
    "sur": "surprise",
    "surprise": "surprise",
    "surprised": "surprise",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return LABEL_ALIASES.get(normalized, normalized)


def label_to_index(label: str) -> int:
    normalized = normalize_label(label)
    if normalized not in CANONICAL_LABELS:
        raise ValueError(f"Unknown canonical emotion label: {label!r}")
    return CANONICAL_LABELS.index(normalized)


def reorder_probabilities(
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, float]:
    if isinstance(probabilities, Mapping):
        values = [float(probabilities[label]) for label in source_labels]
    else:
        values = [float(value) for value in probabilities]

    if len(values) != len(source_labels):
        raise ValueError(
            "Probability count does not match source label count: "
            f"{len(values)} probabilities for {len(source_labels)} labels."
        )

    target_normalized = [normalize_label(label) for label in target_labels]
    unknown_targets = [label for label in target_normalized if label not in CANONICAL_LABELS]
    if unknown_targets:
        raise ValueError(f"Unknown target labels: {unknown_targets}")

    aligned = {label: 0.0 for label in target_normalized}
    seen_sources: set[str] = set()
    for source_label, value in zip(source_labels, values, strict=True):
        normalized = normalize_label(source_label)
        if normalized not in CANONICAL_LABELS:
            raise ValueError(f"Cannot map source label {source_label!r} to a canonical emotion label.")
        if normalized in seen_sources:
            raise ValueError(f"Duplicate source label after normalization: {source_label!r} -> {normalized!r}")
        if normalized not in aligned:
            raise ValueError(f"Source label {source_label!r} maps to {normalized!r}, which is absent from target labels.")
        aligned[normalized] = float(value)
        seen_sources.add(normalized)

    return {label: float(aligned[label]) for label in target_normalized}
