from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import torch

from multimodal_emotion.labels import CANONICAL_LABELS, reorder_logits_to_canonical
from multimodal_emotion.inference.result import PredictionResult
from multimodal_emotion.inference.runtime_config import resolve_model_path


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0.0:
        return np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
    return exp_values / total


CHUNK_POOLING_MODES = {"weighted", "max", "mean"}


def _source_labels_from_id2label(id2label: Mapping | None) -> list[str]:
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("Model config is missing id2label; cannot safely align text labels.")
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


class TextEmotionPredictor:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        temperature: float = 1.0,
        device: str | torch.device | None = "auto",
        chunk_max_tokens: int | None = 48,
        chunk_pooling: str = "weighted",
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path = resolve_model_path("text", model_path)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("Text temperature must be positive.")
        self.device = _resolve_device(device)
        self.chunk_max_tokens = int(chunk_max_tokens) if chunk_max_tokens else 0
        self.chunk_pooling = chunk_pooling if chunk_pooling in CHUNK_POOLING_MODES else "weighted"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.source_labels = _source_labels_from_id2label(getattr(self.model.config, "id2label", None))

    def _max_length(self) -> int:
        tokenizer_limit = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        if 0 < tokenizer_limit < 100_000:
            return tokenizer_limit
        config_limit = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if config_limit > 2:
            return max(1, config_limit - 2)
        return 128

    def _effective_chunk_max_tokens(self) -> int:
        model_limit = max(8, self._max_length())
        configured = int(self.chunk_max_tokens) if self.chunk_max_tokens > 0 else model_limit
        return max(8, min(configured, model_limit))

    def _text_token_count(self, text: str) -> int:
        return max(1, int(len(self.tokenizer.encode(text, add_special_tokens=False))))

    def _split_text_chunks(self, transcript: str) -> list[str]:
        max_tokens = self._effective_chunk_max_tokens()
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|[\n\r]+|,\s+", transcript) if piece.strip()]
        if not pieces:
            pieces = [transcript.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = self._text_token_count(piece)
            if piece_tokens > max_tokens:
                words = piece.split()
                word_buffer: list[str] = []
                split_pieces: list[str] = []
                for word in words:
                    candidate = " ".join([*word_buffer, word])
                    if word_buffer and self._text_token_count(candidate) > max_tokens:
                        split_pieces.append(" ".join(word_buffer))
                        word_buffer = [word]
                    else:
                        word_buffer.append(word)
                if word_buffer:
                    split_pieces.append(" ".join(word_buffer))
            else:
                split_pieces = [piece]

            for split_piece in split_pieces:
                split_tokens = self._text_token_count(split_piece)
                if current and current_tokens + split_tokens > max_tokens:
                    chunks.append(" ".join(current).strip())
                    current = [split_piece]
                    current_tokens = split_tokens
                else:
                    current.append(split_piece)
                    current_tokens += split_tokens

        if current:
            chunks.append(" ".join(current).strip())

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _estimate_speaker_count(text: str) -> int:
        matches = re.findall(r"(?m)^\s*([A-Z][A-Za-z0-9_]{1,20})\s*:", text)
        unique = {match.lower() for match in matches}
        return max(1, len(unique))

    @staticmethod
    def _count_sentences(text: str) -> int:
        sentences = [sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
        return max(1, len(sentences))

    @staticmethod
    def _chunk_weight(token_count: int, confidence: float, exclamation_count: int = 0) -> float:
        token_weight = max(1.0, float(token_count))
        confidence_weight = 0.5 + (0.5 * max(min(float(confidence), 1.0), 0.0))
        exclamation_boost = 1.0 + min(0.30, max(0, int(exclamation_count)) * 0.05)
        return max(token_weight * confidence_weight * exclamation_boost, 0.05)

    @staticmethod
    def _normalize_vector(values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
        return values / total

    @staticmethod
    def _text_quality_multiplier(
        *,
        token_count: int,
        chunk_count: int,
        sentence_count: int,
        speaker_count: int,
        min_quality: float = 0.35,
    ) -> float:
        quality = 1.0
        if token_count >= 160:
            quality *= 0.85
        if token_count >= 280:
            quality *= 0.78
        if token_count >= 420:
            quality *= 0.70

        if sentence_count >= 6:
            quality *= 0.90
        if sentence_count >= 12:
            quality *= 0.82

        if speaker_count >= 2:
            quality *= 0.85
        if speaker_count >= 3:
            quality *= 0.75

        if chunk_count >= 4:
            quality *= 0.90
        if chunk_count >= 6:
            quality *= 0.82

        return max(min(quality, 1.0), min_quality)

    @torch.no_grad()
    def _predict_chunk(self, text: str) -> tuple[np.ndarray, np.ndarray, int]:
        tokenized = self.tokenizer(
            text.strip(),
            truncation=True,
            max_length=self._max_length(),
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        outputs = self.model(**tokenized)
        raw_logits = outputs.logits.detach().cpu().numpy()[0]
        canonical_logits = reorder_logits_to_canonical(raw_logits, self.source_labels)
        probs = _softmax(canonical_logits / self.temperature)
        token_count = int(tokenized["input_ids"].shape[-1])
        return canonical_logits, probs, token_count

    def _pool_chunks(
        self,
        chunk_probs: list[np.ndarray],
        chunk_logits: list[np.ndarray],
        chunk_weights: list[float],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not chunk_probs:
            uniform = np.full(len(CANONICAL_LABELS), 1.0 / len(CANONICAL_LABELS), dtype=np.float64)
            return uniform, None

        probs_stack = np.vstack(chunk_probs)
        logits_stack = np.vstack(chunk_logits) if chunk_logits else None

        if self.chunk_pooling == "max":
            pooled_probs = self._normalize_vector(np.max(probs_stack, axis=0))
            pooled_logits = np.max(logits_stack, axis=0) if logits_stack is not None else None
            return pooled_probs, pooled_logits

        if self.chunk_pooling == "mean":
            pooled_probs = self._normalize_vector(np.mean(probs_stack, axis=0))
            pooled_logits = np.mean(logits_stack, axis=0) if logits_stack is not None else None
            return pooled_probs, pooled_logits

        weights = np.asarray(chunk_weights, dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            weights = np.full(len(chunk_probs), 1.0, dtype=np.float64)
        pooled_probs = self._normalize_vector(np.average(probs_stack, axis=0, weights=weights))
        pooled_logits = np.average(logits_stack, axis=0, weights=weights) if logits_stack is not None else None
        return pooled_probs, pooled_logits

    @torch.no_grad()
    def predict(self, text: str | None) -> PredictionResult:
        result, _ = self.predict_with_chunks(text)
        return result

    @torch.no_grad()
    def predict_with_chunks(self, text: str | None) -> tuple[PredictionResult, list[dict[str, float | str]]]:
        if text is None or not text.strip():
            return (
                PredictionResult.from_unavailable(
                    "text",
                    "Text input is missing or empty.",
                    {"num_chars": 0, "quality_weight_multiplier": 0.0},
                ),
                [],
            )

        cleaned = text.strip()
        chunks = self._split_text_chunks(cleaned)
        if not chunks:
            chunks = [cleaned]

        chunk_probs: list[np.ndarray] = []
        chunk_logits: list[np.ndarray] = []
        chunk_weights: list[float] = []
        chunk_rows: list[dict[str, float | str]] = []
        total_tokens = 0

        for index, chunk in enumerate(chunks, start=1):
            logits, probs, token_count = self._predict_chunk(chunk)
            confidence = float(np.max(probs))
            exclamation_count = chunk.count("!")
            weight = self._chunk_weight(token_count, confidence, exclamation_count)
            total_tokens += token_count
            chunk_probs.append(probs)
            chunk_logits.append(logits)
            chunk_weights.append(weight)

            row: dict[str, float | str] = {
                "chunk_index": index,
                "chunk_text": chunk,
                "num_tokens": float(token_count),
                "weight": float(weight),
                "top_emotion": CANONICAL_LABELS[int(np.argmax(probs))],
                "confidence": float(confidence),
            }
            for label, value in zip(CANONICAL_LABELS, probs, strict=True):
                row[label] = float(value)
            chunk_rows.append(row)

        pooled_probs, pooled_logits = self._pool_chunks(chunk_probs, chunk_logits, chunk_weights)
        pred_idx = int(np.argmax(pooled_probs))

        sentence_count = self._count_sentences(cleaned)
        speaker_count = self._estimate_speaker_count(cleaned)
        quality_multiplier = self._text_quality_multiplier(
            token_count=total_tokens,
            chunk_count=len(chunks),
            sentence_count=sentence_count,
            speaker_count=speaker_count,
        )

        quality = {
            "temperature": self.temperature,
            "num_chars": len(cleaned),
            "num_tokens": total_tokens,
            "chunk_count": len(chunks),
            "chunk_pooling": self.chunk_pooling,
            "sentence_count": sentence_count,
            "speaker_count": speaker_count,
            "quality_weight_multiplier": float(quality_multiplier),
            "chunk_max_tokens": self._effective_chunk_max_tokens(),
        }

        return (
            PredictionResult(
                modality="text",
                available=True,
                labels=list(CANONICAL_LABELS),
                logits=None if pooled_logits is None else [float(value) for value in pooled_logits],
                probs=[float(value) for value in pooled_probs],
                pred_label=CANONICAL_LABELS[pred_idx],
                confidence=float(pooled_probs[pred_idx]),
                quality=quality,
                error=None,
            ),
            chunk_rows,
        )


def predict_text(text: str | None, model_path: str | None = None, temperature: float = 1.0) -> PredictionResult:
    return TextEmotionPredictor(model_path=model_path, temperature=temperature).predict(text)


def predict_text_label(text: str | None, model_path: str | None = None, temperature: float = 1.0) -> str | None:
    return predict_text(text, model_path=model_path, temperature=temperature).pred_label
