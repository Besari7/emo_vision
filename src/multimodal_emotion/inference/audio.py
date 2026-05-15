from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from multimodal_emotion.labels import CANONICAL_LABELS, reorder_logits_to_canonical
from multimodal_emotion.inference.result import PredictionResult
from multimodal_emotion.inference.runtime_config import resolve_model_path


TARGET_SAMPLE_RATE = 16_000


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0:
        return np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
    return exp_values / total


def _source_labels_from_id2label(id2label: Mapping | None) -> list[str]:
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("Model config is missing id2label; cannot safely align audio labels.")
    indexed: list[tuple[int, str]] = []
    for key, value in id2label.items():
        indexed.append((int(key), str(value)))
    return [label for _, label in sorted(indexed)]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


class AudioEmotionPredictor:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        temperature: float = 1.4,
        max_chunk_seconds: float = 6.0,
        chunk_overlap_seconds: float = 0.5,
        min_duration_seconds: float = 0.5,
        short_duration_seconds: float = 1.0,
        silence_rms_threshold: float = 1e-5,
        device: str | torch.device | None = "auto",
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoProcessor

        self.model_path = resolve_model_path("audio", model_path)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("Audio temperature must be positive.")
        self.max_chunk_seconds = float(max_chunk_seconds)
        self.chunk_overlap_seconds = float(chunk_overlap_seconds)
        self.min_duration_seconds = float(min_duration_seconds)
        self.short_duration_seconds = float(short_duration_seconds)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.device = _resolve_device(device)

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.source_labels = _source_labels_from_id2label(getattr(self.model.config, "id2label", None))

    @staticmethod
    def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        import librosa

        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim > 1:
            values = np.mean(values, axis=0 if values.shape[0] <= values.shape[-1] else 1).astype(np.float32)
        values = values.reshape(-1)
        values = values[np.isfinite(values)]
        if sample_rate != TARGET_SAMPLE_RATE and values.size:
            values = librosa.resample(values, orig_sr=int(sample_rate), target_sr=TARGET_SAMPLE_RATE).astype(np.float32)
            sample_rate = TARGET_SAMPLE_RATE
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1.0:
            values = (values / peak).astype(np.float32)
        return values.astype(np.float32), int(sample_rate)

    def _chunk_waveform(self, waveform: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        max_samples = max(1, int(self.max_chunk_seconds * sample_rate))
        overlap_samples = max(0, int(self.chunk_overlap_seconds * sample_rate))
        step = max(1, max_samples - overlap_samples)
        if waveform.shape[0] <= max_samples:
            return [waveform]

        chunks: list[np.ndarray] = []
        start = 0
        while start < waveform.shape[0]:
            end = min(start + max_samples, waveform.shape[0])
            chunks.append(waveform[start:end])
            if end >= waveform.shape[0]:
                break
            start += step
        return chunks

    @torch.no_grad()
    def _predict_chunk_logits(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self.processor(
            chunk.astype(np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, "to")}
        try:
            outputs = self.model(**inputs)
        except RuntimeError as error:
            if self.device.type == "cuda" and "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model.to(self.device)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
            else:
                raise
        raw_logits = outputs.logits.detach().cpu().numpy()[0]
        return reorder_logits_to_canonical(raw_logits, self.source_labels)

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        *,
        source: str | None = None,
    ) -> PredictionResult:
        waveform, sample_rate = self._prepare_waveform(waveform, sample_rate)
        duration_sec = float(waveform.shape[0] / sample_rate) if sample_rate else 0.0
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        base_quality = {
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "rms": rms,
            "source": source,
            "num_chunks": 0,
        }

        if waveform.size == 0:
            return PredictionResult.from_unavailable("audio", "Audio waveform is empty.", base_quality)
        if duration_sec < self.min_duration_seconds:
            return PredictionResult.from_unavailable(
                "audio",
                f"Audio duration is below {self.min_duration_seconds:.2f} seconds.",
                base_quality,
            )
        if rms <= self.silence_rms_threshold:
            return PredictionResult.from_unavailable(
                "audio",
                "Audio RMS is too close to silence.",
                base_quality,
            )

        chunks = self._chunk_waveform(waveform, sample_rate)
        chunk_logits = [self._predict_chunk_logits(chunk, sample_rate) for chunk in chunks if chunk.size]
        if not chunk_logits:
            return PredictionResult.from_unavailable("audio", "No valid audio chunks were available.", base_quality)

        canonical_logits = np.mean(np.stack(chunk_logits, axis=0), axis=0)
        probs = _softmax(canonical_logits / self.temperature)
        pred_idx = int(np.argmax(probs))
        quality_weight = 0.5 if duration_sec < self.short_duration_seconds else 1.0
        return PredictionResult(
            modality="audio",
            available=True,
            labels=list(CANONICAL_LABELS),
            logits=[float(value) for value in canonical_logits],
            probs=[float(value) for value in probs],
            pred_label=CANONICAL_LABELS[pred_idx],
            confidence=float(probs[pred_idx]),
            quality={
                **base_quality,
                "num_chunks": len(chunk_logits),
                "temperature": self.temperature,
                "max_chunk_seconds": self.max_chunk_seconds,
                "quality_weight_multiplier": quality_weight,
            },
            error=None,
        )

    def predict(self, audio_path: str | Path | None) -> PredictionResult:
        if audio_path is None:
            return PredictionResult.from_unavailable("audio", "Audio path is missing.")
        try:
            import librosa

            waveform, sample_rate = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
        except Exception as error:
            return PredictionResult.from_unavailable(
                "audio",
                f"Audio could not be decoded: {error}",
                {"source": str(audio_path), "quality_weight_multiplier": 0.0},
            )
        return self.predict_waveform(waveform, sample_rate, source=str(audio_path))


def predict_audio(audio_path: str | Path | None, model_path: str | None = None, temperature: float = 1.4) -> PredictionResult:
    return AudioEmotionPredictor(model_path=model_path, temperature=temperature).predict(audio_path)
