from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from multimodal_emotion.labels import CANONICAL_LABELS


ModalityName = Literal["text", "audio", "video", "fusion"]


@dataclass(slots=True)
class PredictionResult:
    modality: ModalityName
    available: bool
    labels: list[str] = field(default_factory=lambda: list(CANONICAL_LABELS))
    logits: list[float] | None = None
    probs: list[float] | None = None
    pred_label: str | None = None
    confidence: float = 0.0
    quality: dict = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if list(self.labels) != CANONICAL_LABELS:
            raise ValueError(f"PredictionResult labels must match CANONICAL_LABELS: {self.labels!r}")
        if self.logits is not None and len(self.logits) != len(CANONICAL_LABELS):
            raise ValueError(f"PredictionResult logits must contain {len(CANONICAL_LABELS)} values.")
        if self.probs is not None and len(self.probs) != len(CANONICAL_LABELS):
            raise ValueError(f"PredictionResult probs must contain {len(CANONICAL_LABELS)} values.")
        if self.pred_label is not None and self.pred_label not in CANONICAL_LABELS:
            raise ValueError(f"PredictionResult pred_label is not canonical: {self.pred_label!r}")

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "available": bool(self.available),
            "labels": list(self.labels),
            "logits": None if self.logits is None else [float(value) for value in self.logits],
            "probs": None if self.probs is None else [float(value) for value in self.probs],
            "pred_label": self.pred_label,
            "confidence": float(self.confidence),
            "quality": dict(self.quality),
            "error": self.error,
        }

    @classmethod
    def from_unavailable(
        cls,
        modality: ModalityName,
        error: str,
        quality: dict | None = None,
    ) -> "PredictionResult":
        unavailable_quality = {"quality_weight_multiplier": 0.0}
        if quality:
            unavailable_quality.update(quality)
        return cls(
            modality=modality,
            available=False,
            labels=list(CANONICAL_LABELS),
            logits=None,
            probs=None,
            pred_label=None,
            confidence=0.0,
            quality=unavailable_quality,
            error=error,
        )
