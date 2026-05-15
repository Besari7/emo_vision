from __future__ import annotations

import re
import shutil
import subprocess
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
from transformers import pipeline

from multimodal_emotion.inference import (
    AudioEmotionPredictor,
    FusionEngine,
    PredictionResult,
    TextEmotionPredictor,
    VideoEmotionPredictor,
    load_runtime_config,
)

from .fusion import (
    AUDIO_LABEL_MAP,
    COMMON_LABELS,
    MODALITY_BASE_WEIGHTS,
    TEXT_LABEL_MAP,
    VIDEO_LABEL_MAP,
    ModalitySummary,
    confidence_from_scores,
    remap_predictions,
)


@dataclass(slots=True)
class DemoModelConfig:
    asr_model_name: str = "openai/whisper-small.en"
    sample_frames: int = 16
    hf_audio_model_name: str = "artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop"
    hf_text_model_name: str = "artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7"
    hf_video_model_name: str = "artifacts/video_models/vit_based_fer_model"
    frame_stats_default_max_frames: int = 120
    face_padding_ratio: float = 0.28
    face_smoothing_alpha: float = 0.70
    face_min_size: int = 42
    face_min_relative_area: float = 0.012
    face_center_bias_weight: float = 0.42
    face_area_weight: float = 0.33
    face_temporal_weight: float = 0.25
    face_max_center_jump_ratio: float = 0.32
    face_reacquire_after_misses: int = 6
    face_min_crop_ratio: float = 0.52
    face_max_crop_ratio: float = 0.92
    preprocessed_preview_size: int = 384
    video_only_neutral_reduction_ratio: float = 0.70
    video_disgust_suppression_ratio: float = 0.50
    text_multimodal_weight_reduction_ratio: float = 0.30
    text_chunk_max_tokens: int = 48
    text_chunk_pooling: str = "weighted"
    text_neutral_suppression_enabled: bool = True
    text_neutral_suppression_ratio: float = 0.60
    text_neutral_suppression_min_neutral: float = 0.35
    text_neutral_suppression_min_non_neutral: float = 0.18
    audio_window_seconds: float = 1.6
    audio_hop_seconds: float = 0.8
    audio_min_active_rms: float = 0.006
    audio_dynamic_rms_ratio: float = 0.25
    text_temperature: float = 1.0
    audio_temperature: float = 1.4
    video_temperature: float = 1.3
    video_temporal_smoothing_window: int = 3
    video_missing_face_frame_weight: float = 0.25


class MultimodalDemoAnalyzer:
    def __init__(self, config: DemoModelConfig | None = None) -> None:
        self.runtime_config = load_runtime_config(validate_paths=False)
        if config is None:
            self.config = DemoModelConfig(
                hf_text_model_name=self.runtime_config.model_paths.text,
                hf_audio_model_name=self.runtime_config.model_paths.audio,
                hf_video_model_name=self.runtime_config.model_paths.video,
                text_temperature=self.runtime_config.temperatures["text"],
                audio_temperature=self.runtime_config.temperatures["audio"],
                video_temperature=self.runtime_config.temperatures["video"],
            )
        else:
            self.config = config
        self.pipeline_device = 0 if torch.cuda.is_available() else -1

        self._asr_pipeline = None
        self._hf_text_pipeline = None
        self._hf_audio_pipeline = None
        self._hf_video_pipeline = None
        self._text_predictor = None
        self._audio_predictor = None
        self._video_predictor = None
        self._fusion_engine = FusionEngine(
            global_weights=self.runtime_config.global_weights,
            confidence_gating=self.runtime_config.confidence_gating,
        )
        self._cancel_requested = threading.Event()
        self._face_cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

    def clear_cancel(self) -> None:
        self._cancel_requested.clear()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise RuntimeError("Analysis cancelled by user.")

    @staticmethod
    @contextmanager
    def _runtime_workspace(prefix: str):
        runtime_root = Path.cwd() / "artifacts" / "runtime_workspaces"
        runtime_root.mkdir(parents=True, exist_ok=True)
        workspace = runtime_root / f"{prefix}{uuid.uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=False)
        try:
            yield workspace
        finally:
            # Windows can briefly keep media files locked after OpenCV/ffmpeg/Gradio use.
            # Leftover workspaces are under ignored artifacts/ and should not break the UI.
            shutil.rmtree(workspace, ignore_errors=True)

    def preload_models(self) -> dict[str, str]:
        _ = self.asr_pipeline
        _ = self.text_predictor
        _ = self.audio_predictor
        _ = self.video_predictor
        return {
            "asr": self.config.asr_model_name,
            "text_model": self.config.hf_text_model_name,
            "audio_model": self.config.hf_audio_model_name,
            "video_model": self.config.hf_video_model_name,
        }

    @property
    def asr_pipeline(self):
        if self._asr_pipeline is None:
            model_name = self._resolve_cached_hf_model(self.config.asr_model_name, "ASR")
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                tokenizer=model_name,
                feature_extractor=model_name,
                device=self.pipeline_device,
                chunk_length_s=20,
                generate_kwargs={
                    "temperature": 0.0,
                    "num_beams": 5,
                },
            )
        return self._asr_pipeline

    @property
    def text_predictor(self):
        if self._text_predictor is None:
            self._text_predictor = TextEmotionPredictor(
                model_path=self.config.hf_text_model_name,
                temperature=self.config.text_temperature,
                chunk_max_tokens=self.config.text_chunk_max_tokens,
                chunk_pooling=self.config.text_chunk_pooling,
            )
        return self._text_predictor

    @property
    def audio_predictor(self):
        if self._audio_predictor is None:
            self._audio_predictor = AudioEmotionPredictor(
                model_path=self.config.hf_audio_model_name,
                temperature=self.config.audio_temperature,
            )
        return self._audio_predictor

    @property
    def video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = VideoEmotionPredictor(
                model_path=self.config.hf_video_model_name,
                temperature=self.config.video_temperature,
                sample_frames=self.config.sample_frames,
                face_margin_ratio=self.config.face_padding_ratio,
            )
        return self._video_predictor

    @property
    def hf_text_pipeline(self):
        if self._hf_text_pipeline is None:
            model_name = self._resolve_model_directory(self.config.hf_text_model_name)
            self._hf_text_pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=self.pipeline_device,
                top_k=None,
            )
        return self._hf_text_pipeline

    @property
    def hf_audio_pipeline(self):
        if self._hf_audio_pipeline is None:
            model_name = self._resolve_model_directory(self.config.hf_audio_model_name)
            self._hf_audio_pipeline = pipeline(
                "audio-classification",
                model=model_name,
                device=self.pipeline_device,
                top_k=None,
            )
        return self._hf_audio_pipeline

    @staticmethod
    def _resolve_model_directory(model_name: str) -> str:
        model_path = Path(model_name)
        nested_path = model_path / "best_model"
        has_root_config = (model_path / "config.json").is_file()
        has_root_weights = (model_path / "model.safetensors").is_file() or (model_path / "pytorch_model.bin").is_file()
        has_nested_config = (nested_path / "config.json").is_file()
        has_nested_weights = (nested_path / "model.safetensors").is_file() or (nested_path / "pytorch_model.bin").is_file()
        if model_path.is_dir() and not (has_root_config and has_root_weights) and has_nested_config and has_nested_weights:
            return str(nested_path)
        return model_name

    @staticmethod
    def _resolve_cached_hf_model(model_name: str, label: str) -> str:
        model_path = Path(model_name)
        if model_path.exists():
            return str(model_path)
        if "/" not in model_name:
            return model_name

        try:
            from huggingface_hub import snapshot_download

            return snapshot_download(repo_id=model_name, local_files_only=True)
        except Exception as error:
            raise RuntimeError(
                f"{label} model {model_name!r} is not available in the local Hugging Face cache. "
                "Use the transcript override field or cache the ASR model before automatic transcription."
            ) from error

    @property
    def hf_video_pipeline(self):
        if self._hf_video_pipeline is None:
            self._hf_video_pipeline = pipeline(
                "image-classification",
                model=self.config.hf_video_model_name,
                device=self.pipeline_device,
                top_k=None,
            )
        return self._hf_video_pipeline

    def extract_audio_track(self, video_path: str, output_dir: Path) -> tuple[Path | None, str]:
        audio_path = output_dir / "uploaded_audio.wav"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0 or not audio_path.exists():
            return None, "No audio track detected. The app will continue with the available modalities."
        return audio_path, "Audio track extracted successfully."

    def prepare_browser_safe_video(self, video_path: str | None) -> tuple[str | None, str]:
        if not video_path:
            return None, ""

        source_path = Path(video_path)
        if source_path.name.startswith("emovision_upload_") and source_path.suffix.lower() == ".mp4":
            return str(source_path), "Video preview is ready."
        if not source_path.is_file():
            return video_path, "Uploaded video file could not be found."

        preview_dir = Path.cwd() / "artifacts" / "runtime_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        safe_path = preview_dir / f"emovision_upload_{uuid.uuid4().hex}.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(safe_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0 or not safe_path.exists() or safe_path.stat().st_size == 0:
            return video_path, "Video preview normalization failed; analysis will use the original upload."
        return str(safe_path), "Video converted to browser-safe MP4 preview."

    def transcribe_audio(self, audio_path: Path | None, transcript_override: str | None) -> tuple[str, str, str]:
        if transcript_override and transcript_override.strip():
            return transcript_override.strip(), "manual", "Manual transcript override used."
        if audio_path is None:
            return "", "missing", "No transcript available because the video had no readable audio track."
        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return "", "missing", "ASR skipped transcription because the extracted waveform was empty."

        try:
            transcription = self.asr_pipeline(
                {"array": waveform.astype(np.float32), "sampling_rate": sample_rate}
            )
        except Exception as error:
            return "", "missing", f"ASR model could not be loaded in this environment ({error})."
        transcript = str(transcription.get("text", "")).strip()
        if not transcript:
            return "", "missing", "ASR did not return speech content."
        return transcript, "whisper", "Transcript generated with Whisper ASR."

    def _sample_frames(self, video_path: str) -> list[np.ndarray]:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames: list[np.ndarray] = []

        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num=min(self.config.sample_frames, total_frames), dtype=int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = capture.read()
                if success:
                    sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            while len(sampled_frames) < self.config.sample_frames:
                success, frame = capture.read()
                if not success:
                    break
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        capture.release()
        return sampled_frames

    def _sample_frames_with_indices(self, video_path: str, max_frames: int) -> tuple[list[np.ndarray], list[int]]:
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames: list[np.ndarray] = []
        sampled_indices: list[int] = []

        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, frame = capture.read()
                if success:
                    sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    sampled_indices.append(int(index))
        else:
            index = 0
            while len(sampled_frames) < max_frames:
                success, frame = capture.read()
                if not success:
                    break
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sampled_indices.append(index)
                index += 1

        capture.release()
        return sampled_frames, sampled_indices

    @staticmethod
    def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = box
        return float(x + (w / 2.0)), float(y + (h / 2.0))

    def _detect_face_candidates(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._face_cascade.empty():
            return []
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(self.config.face_min_size, self.config.face_min_size),
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _pick_best_face_candidate(
        self,
        candidates: list[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
        previous_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if not candidates:
            return None

        frame_diag = float(max(np.hypot(frame_width, frame_height), 1.0))
        frame_center = (float(frame_width) / 2.0, float(frame_height) / 2.0)
        prev_center = self._bbox_center(previous_bbox) if previous_bbox is not None else None
        prev_area = float(previous_bbox[2] * previous_bbox[3]) if previous_bbox is not None else 0.0
        frame_area = float(max(frame_width * frame_height, 1))

        best_candidate: tuple[int, int, int, int] | None = None
        best_score = float("-inf")

        for candidate in candidates:
            x, y, w, h = candidate
            area = float(w * h)
            area_ratio = area / frame_area
            cx, cy = self._bbox_center(candidate)

            center_dist = float(np.hypot(cx - frame_center[0], cy - frame_center[1])) / frame_diag
            center_score = 1.0 - min(max(center_dist, 0.0), 1.0)
            area_score = min(max(np.sqrt(max(area_ratio, 0.0)), 0.0), 1.0)

            temporal_score = 0.5
            if prev_center is not None:
                prev_dist = float(np.hypot(cx - prev_center[0], cy - prev_center[1])) / frame_diag
                temporal_score = 1.0 - min(max(prev_dist, 0.0), 1.0)

            score = (
                self.config.face_center_bias_weight * center_score
                + self.config.face_area_weight * area_score
                + self.config.face_temporal_weight * temporal_score
            )

            if area_ratio < self.config.face_min_relative_area:
                score -= 0.25

            if prev_area > 0.0 and area > (prev_area * 3.5):
                score -= 0.10

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    @staticmethod
    def _to_square_bbox(
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
        padding_ratio: float,
        min_crop_ratio: float,
        max_crop_ratio: float,
    ) -> tuple[int, int, int, int]:
        side = int(max(w, h) * (1.0 + padding_ratio))
        frame_min_side = int(min(frame_width, frame_height))
        min_side = int(max(1, frame_min_side * float(min_crop_ratio)))
        max_side = int(max(1, frame_min_side * float(max_crop_ratio)))
        side = max(side, min_side)
        side = min(side, max_side)
        cx = x + (w // 2)
        cy = y + (h // 2)
        x1 = max(cx - (side // 2), 0)
        y1 = max(cy - (side // 2), 0)
        x2 = min(x1 + side, frame_width)
        y2 = min(y1 + side, frame_height)
        side = min(x2 - x1, y2 - y1)
        return int(x1), int(y1), int(side), int(side)

    @staticmethod
    def _smooth_bbox(
        previous_bbox: tuple[int, int, int, int] | None,
        current_bbox: tuple[int, int, int, int],
        alpha: float,
    ) -> tuple[int, int, int, int]:
        if previous_bbox is None:
            return current_bbox
        ax = float(alpha)
        bx = 1.0 - ax
        return (
            int(ax * previous_bbox[0] + bx * current_bbox[0]),
            int(ax * previous_bbox[1] + bx * current_bbox[1]),
            int(ax * previous_bbox[2] + bx * current_bbox[2]),
            int(ax * previous_bbox[3] + bx * current_bbox[3]),
        )

    def _extract_face_focus_frames(
        self,
        frames: list[np.ndarray],
    ) -> tuple[list[np.ndarray], dict]:
        if not frames:
            return [], {"face_detected_frames": 0, "face_ratio": 0.0, "strategy": "none"}

        processed: list[np.ndarray] = []
        previous_bbox: tuple[int, int, int, int] | None = None
        detected_flags: list[bool] = []
        face_detected_count = 0
        consecutive_misses = 0

        for frame in frames:
            height, width = frame.shape[:2]
            detected_candidates = self._detect_face_candidates(frame)
            detected = self._pick_best_face_candidate(
                detected_candidates,
                frame_width=width,
                frame_height=height,
                previous_bbox=previous_bbox,
            )
            if detected is not None and previous_bbox is not None:
                prev_cx, prev_cy = self._bbox_center(previous_bbox)
                cur_cx, cur_cy = self._bbox_center(detected)
                jump = float(np.hypot(cur_cx - prev_cx, cur_cy - prev_cy))
                jump_ratio = jump / float(max(np.hypot(width, height), 1.0))
                if jump_ratio > self.config.face_max_center_jump_ratio and consecutive_misses < 2:
                    detected = None

            if detected is not None:
                x, y, w, h = detected
                candidate_bbox = self._to_square_bbox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    frame_width=width,
                    frame_height=height,
                    padding_ratio=self.config.face_padding_ratio,
                    min_crop_ratio=self.config.face_min_crop_ratio,
                    max_crop_ratio=self.config.face_max_crop_ratio,
                )
                smoothed_bbox = self._smooth_bbox(previous_bbox, candidate_bbox, self.config.face_smoothing_alpha)
                previous_bbox = smoothed_bbox
                face_detected_count += 1
                detected_flags.append(True)
                consecutive_misses = 0
            else:
                detected_flags.append(False)
                consecutive_misses += 1
                if previous_bbox is None or consecutive_misses > self.config.face_reacquire_after_misses:
                    side = int(min(width, height) * 0.70)
                    previous_bbox = ((width - side) // 2, (height - side) // 2, side, side)
                    consecutive_misses = 0

            if previous_bbox is None:
                processed.append(frame)
                continue

            x, y, side, _ = previous_bbox
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            side = max(1, min(side, width - x, height - y))
            crop = frame[y : y + side, x : x + side]
            processed.append(crop if crop.size > 0 else frame)

        total = len(frames)
        return processed, {
            "face_detected_frames": face_detected_count,
            "face_ratio": float(face_detected_count / max(total, 1)),
            "detected_flags": detected_flags,
            "strategy": "haar-tracked-face + temporal scoring + smoothed square crop",
        }

    def _write_preprocessed_preview(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
    ) -> str | None:
        if not frames:
            return None

        preview_side = int(self.config.preprocessed_preview_size)
        preview_dir = Path.cwd() / "artifacts" / "runtime_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = preview_dir / f"mme_pre_frames_{uuid.uuid4().hex}"
        frame_dir.mkdir(parents=True, exist_ok=False)
        preview_path = preview_dir / f"mme_preprocessed_{uuid.uuid4().hex}.mp4"

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not ffmpeg_exe:
            return None

        try:
            for idx, frame in enumerate(frames):
                resized = cv2.resize(frame, (preview_side, preview_side), interpolation=cv2.INTER_AREA)
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                frame_no = frame_indices[idx] if idx < len(frame_indices) else idx
                cv2.putText(
                    bgr,
                    f"Frame: {frame_no}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                frame_path = frame_dir / f"frame_{idx:05d}.png"
                if not cv2.imwrite(str(frame_path), bgr):
                    return None

            command = [
                ffmpeg_exe,
                "-y",
                "-framerate",
                "10",
                "-i",
                str(frame_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(preview_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0 or not preview_path.exists() or preview_path.stat().st_size == 0:
                return None
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

        return str(preview_path)

    def _sample_frames_with_indices_fallback(
        self,
        video_path: str,
        output_dir: Path,
        max_frames: int,
    ) -> tuple[list[np.ndarray], list[int], str]:
        frames, indices = self._sample_frames_with_indices(video_path, max_frames)
        if frames:
            return frames, indices, "Frames were read from the original uploaded video."

        fallback_path, transcode_note = self._transcode_video_for_decoding(video_path, output_dir)
        if fallback_path is None:
            return [], [], transcode_note

        fallback_frames, fallback_indices = self._sample_frames_with_indices(str(fallback_path), max_frames)
        if fallback_frames:
            return fallback_frames, fallback_indices, f"{transcode_note} Frame extraction succeeded on fallback video."
        return [], [], f"{transcode_note} But frame extraction still failed."

    def _transcode_video_for_decoding(self, video_path: str, output_dir: Path) -> tuple[Path | None, str]:
        fallback_path = output_dir / "video_decode_fallback.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            video_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(fallback_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0 or not fallback_path.exists():
            return None, "Frame fallback transcode failed; video decoder could not recover readable frames."
        return fallback_path, "Frame fallback transcoded video to H.264 MP4."

    def _sample_frames_with_fallback(self, video_path: str, output_dir: Path) -> tuple[list[np.ndarray], str]:
        frames = self._sample_frames(video_path)
        if frames:
            return frames, "Frames were read from the original uploaded video."

        fallback_path, transcode_note = self._transcode_video_for_decoding(video_path, output_dir)
        if fallback_path is None:
            return [], transcode_note

        fallback_frames = self._sample_frames(str(fallback_path))
        if fallback_frames:
            return fallback_frames, f"{transcode_note} Frame extraction succeeded on fallback video."
        return [], f"{transcode_note} But frame extraction still failed."

    @staticmethod
    def _normalize_probability_array(values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(COMMON_LABELS), 1.0 / float(len(COMMON_LABELS)), dtype=np.float64)
        return values / total

    @staticmethod
    def _prediction_probabilities(result: PredictionResult) -> dict[str, float]:
        if not result.available or result.probs is None:
            return {label: 0.0 for label in COMMON_LABELS}
        return {label: float(result.probs[index]) for index, label in enumerate(COMMON_LABELS)}

    @staticmethod
    def _summary_from_prediction(result: PredictionResult, note: str | None = None) -> ModalitySummary:
        probabilities = MultimodalDemoAnalyzer._prediction_probabilities(result)
        quality = float(result.quality.get("quality_weight_multiplier", 1.0 if result.available else 0.0))
        status = "ok" if result.available else "missing"
        return ModalitySummary(
            name=result.modality,
            status=status,
            probabilities=probabilities,
            confidence=float(result.confidence),
            quality=quality,
            note=note or result.error or f"{result.modality.title()} model completed.",
        )

    @staticmethod
    def _probabilities_from_logits(logits: np.ndarray, temperature: float) -> dict[str, float]:
        values = np.asarray(logits, dtype=np.float64)
        shifted = (values / float(temperature)) - np.max(values / float(temperature))
        exp_values = np.exp(shifted)
        probabilities = MultimodalDemoAnalyzer._normalize_probability_array(exp_values)
        return {label: float(probabilities[index]) for index, label in enumerate(COMMON_LABELS)}

    @staticmethod
    def _temperature_smooth_probabilities(
        probabilities: dict[str, float],
        temperature: float,
    ) -> dict[str, float]:
        if temperature <= 0.0 or abs(temperature - 1.0) < 1e-6:
            return {label: float(probabilities.get(label, 0.0)) for label in COMMON_LABELS}

        values = np.array([max(float(probabilities.get(label, 0.0)), 1e-12) for label in COMMON_LABELS], dtype=np.float64)
        values = np.power(values, 1.0 / float(temperature))
        values = MultimodalDemoAnalyzer._normalize_probability_array(values)
        return {label: float(values[index]) for index, label in enumerate(COMMON_LABELS)}

    def _smooth_probability_sequence(self, rows: list[dict[str, float]]) -> list[dict[str, float]]:
        window = max(1, int(self.config.video_temporal_smoothing_window))
        if window <= 1 or len(rows) <= 2:
            return rows

        radius = window // 2
        smoothed: list[dict[str, float]] = []
        for index in range(len(rows)):
            start = max(0, index - radius)
            end = min(len(rows), index + radius + 1)
            values = np.array(
                [[float(row.get(label, 0.0)) for label in COMMON_LABELS] for row in rows[start:end]],
                dtype=np.float64,
            )
            averaged = self._normalize_probability_array(values.mean(axis=0))
            smoothed.append({label: float(averaged[label_index]) for label_index, label in enumerate(COMMON_LABELS)})
        return smoothed

    def _text_token_count(self, text: str) -> int:
        tokenizer = getattr(self.hf_text_pipeline, "tokenizer", None)
        if tokenizer is None:
            return max(1, len(text.split()))
        return max(1, len(tokenizer.encode(text, add_special_tokens=False)))

    @staticmethod
    def _text_asr_quality_multiplier(transcript_source: str | None) -> float:
        if transcript_source == "manual":
            return 1.0
        if transcript_source == "whisper":
            return 0.92
        if transcript_source in {"missing", "empty"}:
            return 0.0
        return 0.90

    def _split_text_chunks(self, transcript: str) -> list[str]:
        tokenizer = getattr(self.hf_text_pipeline, "tokenizer", None)
        max_tokens = max(32, int(self.config.text_chunk_max_tokens))
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
                split_pieces = []
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

        if tokenizer is not None:
            return [chunk for chunk in chunks if chunk]
        return [transcript.strip()]

    def _suppress_text_neutral(self, probabilities: dict[str, float]) -> tuple[dict[str, float], bool]:
        normalized = {
            label: float(probabilities.get(label, 0.0))
            for label in COMMON_LABELS
        }
        values = self._normalize_probability_array(
            np.array([normalized[label] for label in COMMON_LABELS], dtype=np.float64)
        )
        normalized = {label: float(values[index]) for index, label in enumerate(COMMON_LABELS)}

        if not self.config.text_neutral_suppression_enabled:
            return normalized, False

        neutral_score = float(normalized.get("neutral", 0.0))
        non_neutral_labels = [label for label in COMMON_LABELS if label != "neutral"]
        non_neutral_mass = float(sum(normalized[label] for label in non_neutral_labels))
        if (
            neutral_score < float(self.config.text_neutral_suppression_min_neutral)
            or non_neutral_mass < float(self.config.text_neutral_suppression_min_non_neutral)
        ):
            return normalized, False

        suppression_ratio = float(np.clip(self.config.text_neutral_suppression_ratio, 0.0, 1.0))
        removed_mass = neutral_score * suppression_ratio
        adjusted = dict(normalized)
        adjusted["neutral"] = neutral_score - removed_mass

        for label in non_neutral_labels:
            adjusted[label] += removed_mass * (normalized[label] / non_neutral_mass)

        adjusted_values = self._normalize_probability_array(
            np.array([adjusted[label] for label in COMMON_LABELS], dtype=np.float64)
        )
        return {label: float(adjusted_values[index]) for index, label in enumerate(COMMON_LABELS)}, removed_mass > 0.0

    def classify_hf_text(self, transcript: str, transcript_source: str | None = None) -> tuple[ModalitySummary, list[dict]]:
        try:
            result, chunk_rows = self.text_predictor.predict_with_chunks(transcript)
        except Exception as error:
            return ModalitySummary(
                name="text",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Text model unavailable ({error}).",
            ), []

        if result.available:
            asr_multiplier = self._text_asr_quality_multiplier(transcript_source)
            existing_quality = float(result.quality.get("quality_weight_multiplier", 1.0))
            result.quality["asr_quality_multiplier"] = float(asr_multiplier)
            result.quality["quality_weight_multiplier"] = float(existing_quality * asr_multiplier)
        else:
            asr_multiplier = 0.0

        summary = self._summary_from_prediction(
            result,
            note=(
                "Text model scored transcript chunks with canonical logit alignment. "
                f"pooling={getattr(self.text_predictor, 'chunk_pooling', 'weighted')}; "
                f"chunks={len(chunk_rows)}; temperature={self.config.text_temperature:.1f}; "
                f"asr_quality={asr_multiplier:.2f}."
                if result.available
                else result.error
            ),
        )
        return summary, chunk_rows

    def _audio_window_ranges(self, total_samples: int, sample_rate: int) -> list[tuple[int, int]]:
        window_samples = max(1, int(self.config.audio_window_seconds * sample_rate))
        hop_samples = max(1, int(self.config.audio_hop_seconds * sample_rate))

        window_ranges: list[tuple[int, int]] = []
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            window_ranges.append((start, end))
            if end >= total_samples:
                break
            start += hop_samples
        return window_ranges

    def _score_active_audio_windows(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[list[dict], list[dict], dict]:
        window_ranges = self._audio_window_ranges(int(waveform.shape[0]), sample_rate)
        rms_values: list[float] = []
        for start_idx, end_idx in window_ranges:
            chunk = waveform[start_idx:end_idx]
            rms_values.append(float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0)

        positive_rms = np.array([value for value in rms_values if value > 0.0], dtype=np.float64)
        dynamic_threshold = (
            float(np.median(positive_rms)) * float(self.config.audio_dynamic_rms_ratio)
            if positive_rms.size
            else 0.0
        )
        active_threshold = max(float(self.config.audio_min_active_rms), dynamic_threshold)

        chart_rows: list[dict] = []
        window_rows: list[dict] = []
        probability_vectors: list[np.ndarray] = []
        top_labels: list[str] = []

        for window_no, (start_idx, end_idx) in enumerate(window_ranges):
            self._raise_if_cancelled()
            chunk = waveform[start_idx:end_idx]
            if chunk.size == 0 or rms_values[window_no] < active_threshold:
                continue
            try:
                result = self.audio_predictor.predict_waveform(chunk.astype(np.float32), sample_rate)
            except Exception as error:
                raise RuntimeError(f"Audio timeline extraction failed ({error}).") from error
            if not result.available or result.probs is None:
                continue

            probs = {label: float(result.probs[index]) for index, label in enumerate(COMMON_LABELS)}
            top_emotion = max(probs, key=probs.get)
            top_conf = float(probs[top_emotion])
            row = {
                "window_no": int(window_no),
                "start_sec": float(start_idx / sample_rate),
                "end_sec": float(end_idx / sample_rate),
                "top_emotion": str(top_emotion.title()),
                "top_confidence": top_conf,
            }
            for label in COMMON_LABELS:
                score = float(probs[label])
                row[label] = score
                chart_rows.append(
                    {
                        "window_no": int(window_no),
                        "emotion": label.title(),
                        "probability": score,
                    }
                )
            window_rows.append(row)
            probability_vectors.append(np.array([probs[label] for label in COMMON_LABELS], dtype=np.float64))
            top_labels.append(top_emotion)

        active_count = len(probability_vectors)
        total_count = len(window_ranges)
        active_ratio = float(active_count / max(total_count, 1))
        if top_labels:
            consistency = max(top_labels.count(label) for label in set(top_labels)) / float(len(top_labels))
        else:
            consistency = 0.0

        meta = {
            "active_count": active_count,
            "total_count": total_count,
            "active_ratio": active_ratio,
            "consistency": float(consistency),
            "active_threshold": float(active_threshold),
            "probability_vectors": probability_vectors,
        }
        return chart_rows, window_rows, meta

    def classify_hf_audio(self, audio_path: Path | None) -> ModalitySummary:
        try:
            result = self.audio_predictor.predict(audio_path)
        except Exception as error:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Audio model unavailable ({error}).",
            )

        return self._summary_from_prediction(
            result,
            note=(
                "Audio model scored 16 kHz waveform chunks with mean-logit aggregation. "
                f"chunks={result.quality.get('num_chunks', 0)}, "
                f"duration={float(result.quality.get('duration_sec', 0.0)):.2f}s, "
                f"rms={float(result.quality.get('rms', 0.0)):.5f}, "
                f"temperature={self.config.audio_temperature:.1f}."
                if result.available
                else result.error
            ),
        )

    def analyze_hf_audio_windows(self, audio_path: Path | None) -> tuple[list[dict], list[dict], str]:
        if audio_path is None:
            return [], [], "Audio timeline unavailable because no audio track exists."

        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return [], [], "Audio timeline unavailable because waveform is empty."

        try:
            chart_rows, window_rows, audio_meta = self._score_active_audio_windows(waveform, sample_rate)
        except Exception as error:
            return [], [], str(error)

        timeline_note = (
            "Audio timeline built from active overlapping windows "
            f"({self.config.audio_window_seconds:.1f}s window / {self.config.audio_hop_seconds:.1f}s hop). "
            f"Active windows: {audio_meta['active_count']}/{audio_meta['total_count']} "
            f"with RMS threshold {audio_meta['active_threshold']:.4f}; "
            f"temperature={self.config.audio_temperature:.1f}."
        )
        return chart_rows, window_rows, timeline_note

    def classify_hf_video(self, video_path: str, output_dir: Path) -> ModalitySummary:
        frames, frame_note = self._sample_frames_with_fallback(video_path, output_dir)
        if not frames:
            return ModalitySummary(
                name="video",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Video model skipped because no frames were readable. {frame_note}",
            )

        face_frames, face_meta = self._extract_face_focus_frames(frames)
        face_ratio = float(face_meta["face_ratio"])
        fallback = bool(self._face_cascade.empty())
        if face_ratio <= 0.0 and not fallback:
            return ModalitySummary(
                name="video",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Video model skipped because no faces were detected. {frame_note}",
            )
        quality = self.config.video_missing_face_frame_weight if fallback else min(max(face_ratio, 0.10), 1.0)
        try:
            result = self.video_predictor.predict_images(
                face_frames,
                fallback=fallback,
                quality={
                    "face_ratio": face_ratio,
                    "face_detected_frames": face_meta["face_detected_frames"],
                    "sampled_frames": len(face_frames),
                    "fallback": fallback,
                    "quality_weight_multiplier": quality,
                },
            )
        except Exception as error:
            return ModalitySummary(
                name="video",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Video model unavailable ({error}).",
            )

        reliability_note = " Low reliability: face detection was intermittent." if face_ratio < 0.75 else ""
        return self._summary_from_prediction(
            result,
            note=(
                "Video model scored face-focused frame crops with temporal smoothing. "
                f"{frame_note} Face-detected frames: {face_meta['face_detected_frames']}/{len(face_frames)} "
                f"({face_ratio * 100.0:.1f}%). "
                f"temperature={self.config.video_temperature:.1f}.{reliability_note}"
                if result.available
                else result.error
            ),
        )

    def analyze_video_frame_statistics(
        self,
        video_path: str,
        max_frames: int | None = None,
        transcript_override: str | None = None,
    ) -> dict:
        with self._runtime_workspace("mme_video_stats_") as workspace:
            self._raise_if_cancelled()
            selected_max_frames = int(max_frames or self.config.frame_stats_default_max_frames)
            selected_max_frames = max(8, min(selected_max_frames, 240))
            audio_path, audio_note = self.extract_audio_track(video_path, workspace)
            self._raise_if_cancelled()
            transcript, transcript_source, transcript_note = self.transcribe_audio(audio_path, transcript_override)
            self._raise_if_cancelled()
            text_modality, text_chunk_rows = self.classify_hf_text(transcript, transcript_source)
            self._raise_if_cancelled()
            audio_modality = self.classify_hf_audio(audio_path)
            self._raise_if_cancelled()
            audio_chart_rows, audio_window_rows, audio_timeline_note = self.analyze_hf_audio_windows(audio_path)
            self._raise_if_cancelled()
            frames, frame_indices, frame_note = self._sample_frames_with_indices_fallback(
                video_path=video_path,
                output_dir=workspace,
                max_frames=selected_max_frames,
            )
            self._raise_if_cancelled()
            if not frames:
                raise ValueError(f"No readable video frames found. {frame_note}")
            face_frames, face_meta = self._extract_face_focus_frames(frames)
            self._raise_if_cancelled()
            preview_video_path = self._write_preprocessed_preview(face_frames, frame_indices)
            self._raise_if_cancelled()

            frame_logits = self.video_predictor.predict_frame_logits(face_frames)
            if not frame_logits:
                raise ValueError("Video model did not return frame logits for the sampled frames.")

            per_frame_rows: list[dict] = []
            chart_rows: list[dict] = []
            video_frame_rows: list[dict] = []
            video_chart_rows: list[dict] = []
            dominance_counts = {label: 0 for label in COMMON_LABELS}
            available_static_modalities: list[tuple[str, dict[str, float], float, float]] = []
            if text_modality.status == "ok":
                available_static_modalities.append(("text", text_modality.probabilities, text_modality.confidence, text_modality.quality))
            if audio_modality.status == "ok":
                available_static_modalities.append(("audio", audio_modality.probabilities, audio_modality.confidence, audio_modality.quality))

            raw_video_rows: list[dict[str, float]] = []
            raw_video_top_rows: list[tuple[str, float]] = []
            for logits in frame_logits:
                video_probs_raw = self._probabilities_from_logits(logits, self.config.video_temperature)
                raw_video_top_rows.append((max(video_probs_raw, key=video_probs_raw.get), float(max(video_probs_raw.values()))))
                raw_video_rows.append(video_probs_raw)

            smoothed_video_rows = self._smooth_probability_sequence(raw_video_rows)
            detected_flags = list(face_meta.get("detected_flags", []))
            face_ratio = float(face_meta["face_ratio"])

            for i, video_probs_for_video_only in enumerate(smoothed_video_rows):
                self._raise_if_cancelled()
                frame_has_face = i < len(detected_flags) and bool(detected_flags[i])
                video_frame_quality = face_ratio if frame_has_face else face_ratio * float(self.config.video_missing_face_frame_weight)
                # Keep Video-Only and Multimodal behavior aligned for video probability adjustments.
                fused_inputs = [
                    ("video", video_probs_for_video_only, confidence_from_scores(video_probs_for_video_only), video_frame_quality),
                    *available_static_modalities,
                ]
                fused_probs, _ = self._weighted_fuse_named(fused_inputs)
                top_emotion = max(fused_probs, key=fused_probs.get)
                dominance_counts[top_emotion] += 1
                raw_top_emotion, raw_top_confidence = raw_video_top_rows[i]
                row = {
                    "frame_no": int(frame_indices[i]),
                    "video_top_emotion": raw_top_emotion,
                    "video_top_confidence": raw_top_confidence,
                    "top_emotion": top_emotion,
                    "top_confidence": float(fused_probs[top_emotion]),
                }
                for label in COMMON_LABELS:
                    row[label] = float(fused_probs[label])
                    chart_rows.append(
                        {
                            "frame_no": int(frame_indices[i]),
                            "emotion": label.title(),
                            "probability": float(fused_probs[label]),
                        }
                    )
                video_row = {
                    "frame_no": int(frame_indices[i]),
                    "top_emotion": max(video_probs_for_video_only, key=video_probs_for_video_only.get),
                    "top_confidence": float(max(video_probs_for_video_only.values())),
                }
                for label in COMMON_LABELS:
                    video_row[label] = float(video_probs_for_video_only[label])
                    video_chart_rows.append(
                        {
                            "frame_no": int(frame_indices[i]),
                            "emotion": label.title(),
                            "probability": float(video_probs_for_video_only[label]),
                        }
                    )
                per_frame_rows.append(row)
                video_frame_rows.append(video_row)

            frame_count = len(per_frame_rows)
            stats_rows: list[dict] = []
            for label in COMMON_LABELS:
                values = np.array([row[label] for row in per_frame_rows], dtype=np.float64)
                wins = int(dominance_counts[label])
                stats_rows.append(
                    {
                        "emotion": label.title(),
                        "mean_probability": float(values.mean()),
                        "max_probability": float(values.max()),
                        "min_probability": float(values.min()),
                        "std_probability": float(values.std()),
                        "dominant_frames": wins,
                        "dominant_ratio_percent": float((wins / frame_count) * 100.0),
                    }
                )

            modality_rows = []
            for modality in [text_modality, audio_modality]:
                top_label = max(modality.probabilities, key=modality.probabilities.get) if any(modality.probabilities.values()) else "unavailable"
                modality_rows.append(
                    {
                        "modality": modality.name.title(),
                        "status": modality.status,
                        "top_emotion": top_label.title(),
                        "confidence": float(modality.confidence),
                        "quality": float(modality.quality),
                        "note": modality.note,
                    }
                )
            text_prob_rows = [
                {"emotion": label.title(), "probability": float(text_modality.probabilities[label])}
                for label in COMMON_LABELS
            ]
            audio_prob_rows = [
                {"emotion": label.title(), "probability": float(audio_modality.probabilities[label])}
                for label in COMMON_LABELS
            ]

            return {
                "frame_count": frame_count,
                "frame_note": frame_note,
                "face_focus_note": (
                    f"Face-focused preprocessing active. Strategy: {face_meta['strategy']}. "
                    f"Detected faces in {face_meta['face_detected_frames']}/{len(face_frames)} frames "
                    f"({face_meta['face_ratio'] * 100.0:.1f}%)."
                ),
                "preprocessed_video_path": preview_video_path,
                "chart_rows": chart_rows,
                "per_frame_rows": per_frame_rows,
                "video_chart_rows": video_chart_rows,
                "video_frame_rows": video_frame_rows,
                "audio_chart_rows": audio_chart_rows,
                "audio_window_rows": audio_window_rows,
                "stats_rows": stats_rows,
                "modality_rows": modality_rows,
                "text_prob_rows": text_prob_rows,
                "text_chunk_rows": text_chunk_rows,
                "audio_prob_rows": audio_prob_rows,
                "transcript": transcript,
                "transcript_source": transcript_source,
                "audio_note": audio_note,
                "audio_timeline_note": audio_timeline_note,
                "transcript_note": transcript_note,
                "video_model_path": self.config.hf_video_model_name,
            }

    def _rebalance_video_only_neutral(self, probabilities: dict[str, float]) -> dict[str, float]:
        neutral_label = "neutral"
        if neutral_label not in probabilities:
            return {label: float(probabilities.get(label, 0.0)) for label in COMMON_LABELS}

        reduction_ratio = float(np.clip(self.config.video_only_neutral_reduction_ratio, 0.0, 1.0))
        neutral_score = float(probabilities.get(neutral_label, 0.0))
        removed_mass = neutral_score * reduction_ratio
        kept_neutral = neutral_score - removed_mass

        non_neutral_labels = [label for label in COMMON_LABELS if label != neutral_label]
        redistributed = removed_mass / float(len(non_neutral_labels)) if non_neutral_labels else 0.0

        adjusted: dict[str, float] = {}
        for label in COMMON_LABELS:
            base_score = float(probabilities.get(label, 0.0))
            if label == neutral_label:
                adjusted[label] = kept_neutral
            else:
                adjusted[label] = base_score + redistributed

        total = float(sum(adjusted.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(COMMON_LABELS))
            return {label: uniform for label in COMMON_LABELS}
        return {label: float(score / total) for label, score in adjusted.items()}

    def _suppress_video_disgust(self, probabilities: dict[str, float]) -> dict[str, float]:
        disgust_label = "disgust"
        if disgust_label not in probabilities:
            return {label: float(probabilities.get(label, 0.0)) for label in COMMON_LABELS}

        suppression_ratio = float(np.clip(self.config.video_disgust_suppression_ratio, 0.0, 1.0))
        disgust_score = float(probabilities.get(disgust_label, 0.0))
        removed_mass = disgust_score * suppression_ratio
        kept_disgust = disgust_score - removed_mass

        target_labels = [label for label in COMMON_LABELS if label != disgust_label]
        redistributed = removed_mass / float(len(target_labels)) if target_labels else 0.0

        adjusted: dict[str, float] = {}
        for label in COMMON_LABELS:
            base_score = float(probabilities.get(label, 0.0))
            adjusted[label] = kept_disgust if label == disgust_label else base_score + redistributed

        total = float(sum(adjusted.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(COMMON_LABELS))
            return {label: uniform for label in COMMON_LABELS}
        return {label: float(score / total) for label, score in adjusted.items()}

    def _adjust_video_probabilities(self, probabilities: dict[str, float]) -> dict[str, float]:
        adjusted = self._rebalance_video_only_neutral(probabilities)
        return self._suppress_video_disgust(adjusted)

    def _build_multimodal_weights(
        self,
        modality_names: list[str],
        reliability: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        if not modality_names:
            return {}

        unique_names = list(dict.fromkeys(modality_names))
        weights = {name: float(MODALITY_BASE_WEIGHTS.get(name, 1.0)) for name in unique_names}
        reduction_ratio = float(np.clip(self.config.text_multimodal_weight_reduction_ratio, 0.0, 1.0))
        other_names = [name for name in unique_names if name != "text"]
        if "text" in weights and other_names and reduction_ratio > 0.0:
            removed_weight = weights["text"] * reduction_ratio
            weights["text"] = weights["text"] - removed_weight
            redistribution = removed_weight / float(len(other_names))
            for name in other_names:
                weights[name] += redistribution

        reliability = reliability or {}
        for name in unique_names:
            confidence, quality = reliability.get(name, (1.0, 1.0))
            weights[name] *= max(float(confidence), 0.05) * max(float(quality), 0.05)

        total = float(sum(weights.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(unique_names))
            return {name: uniform for name in unique_names}
        return {name: weight / total for name, weight in weights.items()}

    def _weighted_fuse_named(
        self,
        named_probability_sets: list[tuple],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not named_probability_sets:
            uniform = {label: 1.0 / len(COMMON_LABELS) for label in COMMON_LABELS}
            return uniform, {}

        predictions: list[PredictionResult] = []
        for item in named_probability_sets:
            name = item[0]
            probabilities = item[1]
            confidence = float(item[2]) if len(item) >= 3 else confidence_from_scores(probabilities)
            quality = float(item[3]) if len(item) >= 4 else 1.0
            values = np.array([float(probabilities.get(label, 0.0)) for label in COMMON_LABELS], dtype=np.float64)
            values = self._normalize_probability_array(values)
            predictions.append(
                PredictionResult(
                    modality=name,
                    available=True,
                    labels=list(COMMON_LABELS),
                    logits=None,
                    probs=[float(value) for value in values],
                    pred_label=COMMON_LABELS[int(np.argmax(values))],
                    confidence=confidence,
                    quality={"quality_weight_multiplier": quality},
                    error=None,
                )
            )

        result = self._fusion_engine.fuse(predictions)
        if not result.available or result.probs is None:
            uniform = {label: 1.0 / len(COMMON_LABELS) for label in COMMON_LABELS}
            return uniform, {}
        normalized = {label: float(result.probs[index]) for index, label in enumerate(COMMON_LABELS)}
        return normalized, dict(result.quality.get("weights_used", {}))

    @staticmethod
    def _modality_rows(modalities: list[ModalitySummary], fusion_weights: dict[str, float]) -> list[dict]:
        rows: list[dict] = []
        for modality in modalities:
            top_label = (
                max(modality.probabilities, key=modality.probabilities.get).title()
                if any(modality.probabilities.values())
                else "Unavailable"
            )
            rows.append(
                {
                    "modality": modality.name.title(),
                    "status": modality.status,
                    "top_emotion": top_label,
                    "confidence": round(modality.confidence, 4),
                    "quality": round(modality.quality, 4),
                    "fusion_weight": round(float(fusion_weights.get(modality.name, 0.0)), 4),
                    "note": modality.note,
                }
            )
        return rows

    def analyze(self, video_path: str, transcript_override: str | None = None) -> dict:
        with self._runtime_workspace("mme_demo_") as workspace:
            self._raise_if_cancelled()
            audio_path, audio_note = self.extract_audio_track(video_path, workspace)
            self._raise_if_cancelled()
            transcript, transcript_source, transcript_note = self.transcribe_audio(audio_path, transcript_override)
            self._raise_if_cancelled()

            text_modality, text_chunk_rows = self.classify_hf_text(transcript, transcript_source)
            modalities: list[ModalitySummary] = [
                text_modality,
                self.classify_hf_video(video_path, workspace),
            ]
            self._raise_if_cancelled()
            modalities.append(self.classify_hf_audio(audio_path))
            self._raise_if_cancelled()

            available_probability_sets: list[tuple[str, dict[str, float], float, float]] = []
            for modality in modalities:
                if modality.status == "ok":
                    available_probability_sets.append(
                        (modality.name, modality.probabilities, modality.confidence, modality.quality)
                    )

            blended_probabilities, fusion_weights = self._weighted_fuse_named(available_probability_sets)
            predicted_label = max(blended_probabilities, key=blended_probabilities.get)
            confidence = float(blended_probabilities[predicted_label])

            probability_rows = [
                {
                    "emotion": label.title(),
                    "score": round(blended_probabilities[label], 4),
                }
                for label in COMMON_LABELS
            ]

            modality_rows = self._modality_rows(modalities, fusion_weights)

            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "probabilities": blended_probabilities,
                "probability_rows": probability_rows,
                "modality_rows": modality_rows,
                "text_chunk_rows": text_chunk_rows,
                "transcript": transcript,
                "transcript_source": transcript_source,
                "fusion_weights": {
                    "text": float(fusion_weights.get("text", 0.0)),
                    "audio": float(fusion_weights.get("audio", 0.0)),
                    "video": float(fusion_weights.get("video", 0.0)),
                },
                "available_modalities": [item[0] for item in available_probability_sets],
                "notes": {
                    "audio": audio_note,
                    "transcript": transcript_note,
                    "fusion": "Fusion uses canonical probabilities, quality multipliers, and renormalized modality weights.",
                },
            }
