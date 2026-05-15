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
    normalized = str(label).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return LABEL_ALIASES.get(normalized, normalized)


label_to_id = {label: index for index, label in enumerate(CANONICAL_LABELS)}
id_to_label = {index: label for index, label in enumerate(CANONICAL_LABELS)}


def label_to_index(label: str) -> int:
    normalized = normalize_label(label)
    if normalized not in label_to_id:
        raise ValueError(f"Unknown canonical emotion label: {label!r}")
    return label_to_id[normalized]


def validate_label_set(source_labels: Sequence[str]) -> list[str]:
    normalized_labels = [normalize_label(label) for label in source_labels]
    if len(normalized_labels) != len(CANONICAL_LABELS):
        raise ValueError(
            "Model output label count must be exactly "
            f"{len(CANONICAL_LABELS)}; received {len(normalized_labels)} labels: {list(source_labels)!r}."
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    unknown: list[str] = []
    for raw_label, normalized in zip(source_labels, normalized_labels, strict=True):
        if normalized not in label_to_id:
            unknown.append(str(raw_label))
            continue
        if normalized in seen:
            duplicates.append(str(raw_label))
        seen.add(normalized)

    if unknown:
        raise ValueError(f"Unknown model labels after alias normalization: {unknown}.")
    if duplicates:
        raise ValueError(f"Duplicate model labels after alias normalization: {duplicates}.")

    missing = [label for label in CANONICAL_LABELS if label not in seen]
    if missing:
        raise ValueError(f"Model labels are missing canonical labels: {missing}.")

    return normalized_labels


def _coerce_vector(values: Mapping[str, float] | Sequence[float] | np.ndarray, source_labels: Sequence[str], name: str) -> np.ndarray:
    if isinstance(values, Mapping):
        vector = np.asarray([float(values[label]) for label in source_labels], dtype=np.float64)
    else:
        vector = np.asarray(values, dtype=np.float64)

    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector; received shape {vector.shape}.")
    if vector.shape[0] != len(CANONICAL_LABELS):
        raise ValueError(
            f"{name} vector length must be exactly {len(CANONICAL_LABELS)}; "
            f"received {vector.shape[0]}."
        )
    if len(source_labels) != len(CANONICAL_LABELS):
        raise ValueError(
            "Source label count must be exactly "
            f"{len(CANONICAL_LABELS)}; received {len(source_labels)}."
        )
    return vector


def _reorder_vector_to_canonical(
    values: Mapping[str, float] | Sequence[float] | np.ndarray,
    source_labels: Sequence[str],
    name: str,
) -> np.ndarray:
    vector = _coerce_vector(values, source_labels, name)
    normalized_labels = validate_label_set(source_labels)
    source_index_by_label = {label: index for index, label in enumerate(normalized_labels)}
    return np.asarray(
        [float(vector[source_index_by_label[label]]) for label in CANONICAL_LABELS],
        dtype=np.float64,
    )


def reorder_probs_to_canonical(
    probs: Mapping[str, float] | Sequence[float] | np.ndarray,
    source_labels: Sequence[str],
) -> np.ndarray:
    return _reorder_vector_to_canonical(probs, source_labels, "Probability")


def reorder_logits_to_canonical(
    logits: Mapping[str, float] | Sequence[float] | np.ndarray,
    source_labels: Sequence[str],
) -> np.ndarray:
    return _reorder_vector_to_canonical(logits, source_labels, "Logit")


def reorder_probabilities(
    probabilities: Mapping[str, float] | Sequence[float] | np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, float]:
    target_normalized = [normalize_label(label) for label in target_labels]
    if target_normalized != CANONICAL_LABELS:
        raise ValueError(
            "Only canonical target label order is supported for probability reordering. "
            f"Received {target_labels!r}."
        )
    aligned = reorder_probs_to_canonical(probabilities, source_labels)
    return {label: float(aligned[index]) for index, label in enumerate(CANONICAL_LABELS)}
