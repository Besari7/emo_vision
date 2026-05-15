from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

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


def _source_labels_from_id2label(id2label: Mapping | None) -> list[str]:
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("Model config is missing id2label; cannot safely align video labels.")
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


class VideoEmotionPredictor:
    def __init__(
        self,
        model_path: str | None = None,
        *,
        temperature: float = 1.3,
        sample_frames: int = 16,
        face_margin_ratio: float = 0.28,
        center_fallback_weight: float = 0.5,
        device: str | torch.device | None = "auto",
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_path = resolve_model_path("video", model_path)
        self.temperature = float(temperature)
        if self.temperature <= 0.0:
            raise ValueError("Video temperature must be positive.")
        self.sample_frames = int(sample_frames)
        self.face_margin_ratio = float(face_margin_ratio)
        self.center_fallback_weight = float(center_fallback_weight)
        self.device = _resolve_device(device)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.source_labels = _source_labels_from_id2label(getattr(self.model.config, "id2label", None))

        self.face_cascade = None
        try:
            import cv2

            cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
            if not cascade.empty():
                self.face_cascade = cascade
        except Exception:
            self.face_cascade = None

    def sample_video_frames(self, video_path: str | Path) -> list[np.ndarray]:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return []
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[np.ndarray] = []
        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num=min(self.sample_frames, total_frames), dtype=int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = capture.read()
                if success:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            while len(frames) < self.sample_frames:
                success, frame = capture.read()
                if not success:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return frames

    @staticmethod
    def _center_crop(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        side = max(1, min(height, width))
        x1 = max(0, (width - side) // 2)
        y1 = max(0, (height - side) // 2)
        return frame[y1 : y1 + side, x1 : x1 + side]

    def _face_crop(self, frame: np.ndarray) -> np.ndarray | None:
        if self.face_cascade is None:
            return None

        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(42, 42))
        if len(faces) == 0:
            return None

        height, width = frame.shape[:2]
        frame_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
        best_box = None
        best_score = float("-inf")
        for x, y, w, h in faces:
            area = float(w * h)
            center = np.asarray([x + (w / 2.0), y + (h / 2.0)], dtype=np.float64)
            center_dist = float(np.linalg.norm(center - frame_center) / max(np.linalg.norm(frame_center), 1.0))
            score = area * (1.0 - min(center_dist, 1.0))
            if score > best_score:
                best_score = score
                best_box = (int(x), int(y), int(w), int(h))

        if best_box is None:
            return None

        x, y, w, h = best_box
        side = int(max(w, h) * (1.0 + self.face_margin_ratio))
        cx = x + (w // 2)
        cy = y + (h // 2)
        x1 = max(0, cx - (side // 2))
        y1 = max(0, cy - (side // 2))
        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)
        side = min(x2 - x1, y2 - y1)
        if side <= 0:
            return None
        return frame[y1 : y1 + side, x1 : x1 + side]

    def face_crops(self, frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], dict]:
        if not frames:
            return [], {"fallback": False, "face_frames": 0, "sampled_frames": 0, "face_ratio": 0.0}
        if self.face_cascade is None:
            return [self._center_crop(frame) for frame in frames], {
                "fallback": True,
                "face_frames": 0,
                "sampled_frames": len(frames),
                "face_ratio": 0.0,
                "quality_weight_multiplier": self.center_fallback_weight,
            }

        crops: list[np.ndarray] = []
        for frame in frames:
            crop = self._face_crop(frame)
            if crop is not None:
                crops.append(crop)
        face_ratio = float(len(crops) / max(len(frames), 1))
        return crops, {
            "fallback": False,
            "face_frames": len(crops),
            "sampled_frames": len(frames),
            "face_ratio": face_ratio,
            "quality_weight_multiplier": face_ratio,
        }

    @staticmethod
    def _to_pil_images(images: Sequence) -> list:
        from PIL import Image

        pil_images = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            else:
                pil_images.append(Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB"))
        return pil_images

    @torch.no_grad()
    def predict_frame_logits(self, images: Sequence) -> list[np.ndarray]:
        if not images:
            return []
        pil_images = self._to_pil_images(images)
        inputs = self.image_processor(pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items() if hasattr(value, "to")}
        try:
            outputs = self.model(**inputs)
            raw_logits = outputs.logits.detach().cpu().numpy()
        except RuntimeError as error:
            if self.device.type == "cuda" and "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                rows: list[np.ndarray] = []
                for image in pil_images:
                    single_inputs = self.image_processor(image, return_tensors="pt")
                    single_inputs = {key: value.to(self.device) for key, value in single_inputs.items() if hasattr(value, "to")}
                    single_outputs = self.model(**single_inputs)
                    rows.append(single_outputs.logits.detach().cpu().numpy()[0])
                raw_logits = np.stack(rows, axis=0)
            else:
                raise
        return [reorder_logits_to_canonical(row, self.source_labels) for row in raw_logits]

    def predict_images(
        self,
        images: Sequence,
        *,
        fallback: bool = False,
        quality: dict | None = None,
    ) -> PredictionResult:
        quality = dict(quality or {})
        quality.setdefault("fallback", bool(fallback))
        if not images:
            return PredictionResult.from_unavailable("video", "No valid video frames were available.", quality)

        frame_logits = self.predict_frame_logits(images)
        if not frame_logits:
            return PredictionResult.from_unavailable("video", "No frame logits were produced.", quality)

        canonical_logits = np.mean(np.stack(frame_logits, axis=0), axis=0)
        probs = _softmax(canonical_logits / self.temperature)
        pred_idx = int(np.argmax(probs))
        quality.setdefault("num_frames", len(frame_logits))
        quality.setdefault("temperature", self.temperature)
        quality.setdefault("quality_weight_multiplier", self.center_fallback_weight if fallback else 1.0)
        return PredictionResult(
            modality="video",
            available=True,
            labels=list(CANONICAL_LABELS),
            logits=[float(value) for value in canonical_logits],
            probs=[float(value) for value in probs],
            pred_label=CANONICAL_LABELS[pred_idx],
            confidence=float(probs[pred_idx]),
            quality=quality,
            error=None,
        )

    def predict(self, video_path: str | Path | None) -> PredictionResult:
        if video_path is None:
            return PredictionResult.from_unavailable("video", "Video path is missing.")
        frames = self.sample_video_frames(video_path)
        if not frames:
            return PredictionResult.from_unavailable(
                "video",
                "Video could not be decoded or no frames were readable.",
                {"source": str(video_path), "quality_weight_multiplier": 0.0},
            )
        crops, quality = self.face_crops(frames)
        quality["source"] = str(video_path)
        if not crops:
            return PredictionResult.from_unavailable(
                "video",
                "No valid face crops were detected.",
                quality,
            )
        return self.predict_images(crops, fallback=bool(quality.get("fallback", False)), quality=quality)


def predict_video(video_path: str | Path | None, model_path: str | None = None, temperature: float = 1.3) -> PredictionResult:
    return VideoEmotionPredictor(model_path=model_path, temperature=temperature).predict(video_path)
