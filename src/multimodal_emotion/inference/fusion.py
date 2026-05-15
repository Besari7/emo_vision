from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from multimodal_emotion.labels import CANONICAL_LABELS
from multimodal_emotion.inference.result import PredictionResult
from multimodal_emotion.inference.runtime_config import (
    ConfidenceGatingConfig,
    CONFIDENCE_GATING,
    GLOBAL_WEIGHTS,
)


FUSION_MODES = {"weighted_probs", "log_probs"}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0:
        return np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
    return exp_values / total


def _coerce_confidence_gating(value: ConfidenceGatingConfig | dict | None) -> ConfidenceGatingConfig:
    if isinstance(value, ConfidenceGatingConfig):
        return value
    values = dict(CONFIDENCE_GATING)
    if isinstance(value, dict):
        if "enabled" in value:
            values["enabled"] = bool(value["enabled"])
        for key in ("video_min_conf_zero", "video_min_conf_low", "video_low_multiplier"):
            if value.get(key) is not None:
                values[key] = float(value[key])
    return ConfidenceGatingConfig(
        enabled=bool(values["enabled"]),
        video_min_conf_zero=float(values["video_min_conf_zero"]),
        video_min_conf_low=float(values["video_min_conf_low"]),
        video_low_multiplier=float(values["video_low_multiplier"]),
    )


class FusionEngine:
    def __init__(
        self,
        global_weights: dict[str, float] | None = None,
        *,
        mode: str = "weighted_probs",
        eps: float = 1e-12,
        confidence_gating: ConfidenceGatingConfig | dict | None = None,
    ) -> None:
        if mode not in FUSION_MODES:
            raise ValueError(f"Unsupported fusion mode {mode!r}. Expected one of {sorted(FUSION_MODES)}.")
        self.global_weights = dict(global_weights or GLOBAL_WEIGHTS)
        self.mode = mode
        self.eps = float(eps)
        self.confidence_gating = _coerce_confidence_gating(confidence_gating)

    @staticmethod
    def _quality_multiplier(prediction: PredictionResult) -> float:
        value = prediction.quality.get("quality_weight_multiplier", 1.0)
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return 1.0

    def _confidence_multiplier(self, prediction: PredictionResult) -> float:
        if not self.confidence_gating.enabled:
            return 1.0
        if prediction.modality != "video":
            return 1.0

        confidence = max(float(prediction.confidence), 0.0)
        if confidence < self.confidence_gating.video_min_conf_zero:
            return 0.0
        if confidence < self.confidence_gating.video_min_conf_low:
            return min(max(float(self.confidence_gating.video_low_multiplier), 0.0), 1.0)
        return 1.0

    def _effective_weight(self, prediction: PredictionResult) -> float:
        base_weight = max(float(self.global_weights.get(prediction.modality, 0.0)), 0.0)
        return base_weight * self._quality_multiplier(prediction) * self._confidence_multiplier(prediction)

    def _available_predictions(self, predictions: Iterable[PredictionResult]) -> list[PredictionResult]:
        available: list[PredictionResult] = []
        for prediction in predictions:
            if prediction.modality == "fusion":
                continue
            if not prediction.available or prediction.probs is None:
                continue
            probs = np.asarray(prediction.probs, dtype=np.float64)
            if probs.shape != (len(CANONICAL_LABELS),):
                raise ValueError(f"{prediction.modality} prediction does not contain 7 canonical probabilities.")
            if self._effective_weight(prediction) <= 0.0:
                continue
            available.append(prediction)
        return available

    def _normalized_weights(self, predictions: list[PredictionResult]) -> dict[str, float]:
        raw: dict[str, float] = {}
        for prediction in predictions:
            raw_weight = self._effective_weight(prediction)
            if raw_weight > 0.0:
                raw[prediction.modality] = raw_weight
        total = float(sum(raw.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(predictions))
            return {prediction.modality: uniform for prediction in predictions}
        return {name: weight / total for name, weight in raw.items()}

    def fuse(self, predictions: Iterable[PredictionResult]) -> PredictionResult:
        available = self._available_predictions(predictions)
        if not available:
            return PredictionResult.from_unavailable(
                "fusion",
                "No available modalities for fusion.",
                {"weights_used": {}, "mode": self.mode},
            )

        weights = self._normalized_weights(available)
        if len(available) == 1:
            single = available[0]
            probs = np.asarray(single.probs, dtype=np.float64)
            pred_idx = int(np.argmax(probs))
            return PredictionResult(
                modality="fusion",
                available=True,
                labels=list(CANONICAL_LABELS),
                logits=single.logits,
                probs=[float(value) for value in probs],
                pred_label=CANONICAL_LABELS[pred_idx],
                confidence=float(probs[pred_idx]),
                quality={
                    "weights_used": {single.modality: 1.0},
                    "mode": self.mode,
                    "single_modality": single.modality,
                    "confidence_gating_enabled": self.confidence_gating.enabled,
                },
                error=None,
            )

        if self.mode == "weighted_probs":
            final_scores = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
            for prediction in available:
                final_scores += float(weights[prediction.modality]) * np.asarray(prediction.probs, dtype=np.float64)

            total = float(final_scores.sum())
            if total <= 0.0:
                final_probs = np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
            else:
                final_probs = final_scores / total
        else:
            log_scores = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
            for prediction in available:
                probs = np.asarray(prediction.probs, dtype=np.float64)
                log_scores += float(weights[prediction.modality]) * np.log(probs + self.eps)
            final_probs = _softmax(log_scores)

        pred_idx = int(np.argmax(final_probs))
        return PredictionResult(
            modality="fusion",
            available=True,
            labels=list(CANONICAL_LABELS),
            logits=None,
            probs=[float(value) for value in final_probs],
            pred_label=CANONICAL_LABELS[pred_idx],
            confidence=float(final_probs[pred_idx]),
            quality={
                "weights_used": {name: float(weight) for name, weight in weights.items()},
                "mode": self.mode,
                "confidence_gating_enabled": self.confidence_gating.enabled,
            },
            error=None,
        )


def fuse_predictions(
    predictions: Iterable[PredictionResult],
    global_weights: dict[str, float] | None = None,
    *,
    mode: str = "weighted_probs",
    confidence_gating: ConfidenceGatingConfig | dict | None = None,
) -> PredictionResult:
    return FusionEngine(
        global_weights=global_weights,
        mode=mode,
        confidence_gating=confidence_gating,
    ).fuse(predictions)
