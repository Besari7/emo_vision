from __future__ import annotations

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

from .fusion import (
    AUDIO_LABEL_MAP,
    COMMON_LABELS,
    TEXT_LABEL_MAP,
    VIDEO_LABEL_MAP,
    ModalitySummary,
    confidence_from_scores,
    remap_predictions,
)


@dataclass(slots=True)
class DemoModelConfig:
    asr_model_name: str = "openai/whisper-tiny.en"
    sample_frames: int = 8
    hf_audio_model_name: str = "artifacts/audio_models/wav2vec2_ravdess_7class"
    hf_text_model_name: str = "artifacts/text_models/roberta_large_goemotions_v2_clean_es"
    hf_video_model_name: str = "artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition"
    frame_stats_default_max_frames: int = 120
    face_padding_ratio: float = 0.28
    face_smoothing_alpha: float = 0.70
    face_min_size: int = 42
    preprocessed_preview_size: int = 384
    video_only_neutral_reduction_ratio: float = 0.70
    video_disgust_suppression_ratio: float = 0.50
    text_multimodal_weight_reduction_ratio: float = 0.30
    audio_window_seconds: float = 1.6
    audio_hop_seconds: float = 0.8


class MultimodalDemoAnalyzer:
    def __init__(self, config: DemoModelConfig | None = None) -> None:
        self.config = config or DemoModelConfig()
        self.pipeline_device = 0 if torch.cuda.is_available() else -1

        self._asr_pipeline = None
        self._hf_text_pipeline = None
        self._hf_audio_pipeline = None
        self._hf_video_pipeline = None
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
        _ = self.hf_text_pipeline
        _ = self.hf_audio_pipeline
        _ = self.hf_video_pipeline
        return {
            "asr": self.config.asr_model_name,
            "text_model": self.config.hf_text_model_name,
            "audio_model": self.config.hf_audio_model_name,
            "video_model": self.config.hf_video_model_name,
        }

    @property
    def asr_pipeline(self):
        if self._asr_pipeline is None:
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.config.asr_model_name,
                device=self.pipeline_device,
                chunk_length_s=20,
            )
        return self._asr_pipeline

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

    def _detect_primary_face(self, frame_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        if self._face_cascade.empty():
            return None
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.config.face_min_size, self.config.face_min_size),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _to_square_bbox(
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        side = int(max(w, h) * (1.0 + padding_ratio))
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
        face_detected_count = 0

        for frame in frames:
            height, width = frame.shape[:2]
            detected = self._detect_primary_face(frame)
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
                )
                smoothed_bbox = self._smooth_bbox(previous_bbox, candidate_bbox, self.config.face_smoothing_alpha)
                previous_bbox = smoothed_bbox
                face_detected_count += 1
            elif previous_bbox is None:
                side = int(min(width, height) * 0.70)
                previous_bbox = ((width - side) // 2, (height - side) // 2, side, side)

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
            "strategy": "haar-primary-face + smoothed square crop",
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

    def classify_hf_text(self, transcript: str) -> ModalitySummary:
        if not transcript.strip():
            return ModalitySummary(
                name="text",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="Text model skipped because transcript is empty.",
            )
        try:
            predictions = self.hf_text_pipeline(transcript)
        except Exception as error:
            return ModalitySummary(
                name="text",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Text model unavailable ({error}).",
            )
        prediction_list = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        probabilities = remap_predictions(prediction_list, TEXT_LABEL_MAP)
        confidence = confidence_from_scores(probabilities)
        quality = min(max(len(transcript.split()) / 14.0, 0.35), 1.0)
        return ModalitySummary(
            name="text",
            status="ok",
            probabilities=probabilities,
            confidence=confidence,
            quality=quality,
            note="Text model scored the transcript.",
        )

    def classify_hf_audio(self, audio_path: Path | None) -> ModalitySummary:
        if audio_path is None:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="Audio model skipped because no audio track exists.",
            )

        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note="Audio model skipped because waveform is empty.",
            )

        try:
            predictions = self.hf_audio_pipeline({"array": waveform, "sampling_rate": sample_rate})
        except Exception as error:
            return ModalitySummary(
                name="audio",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Audio model unavailable ({error}).",
            )

        prediction_list = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
        probabilities = remap_predictions(prediction_list, AUDIO_LABEL_MAP)
        confidence = confidence_from_scores(probabilities)
        duration_seconds = waveform.shape[0] / float(sample_rate)
        quality = min(max(duration_seconds / 6.0, 0.30), 1.0)
        return ModalitySummary(
            name="audio",
            status="ok",
            probabilities=probabilities,
            confidence=confidence,
            quality=quality,
            note="Audio model scored vocal emotion.",
        )

    def analyze_hf_audio_windows(self, audio_path: Path | None) -> tuple[list[dict], list[dict], str]:
        if audio_path is None:
            return [], [], "Audio timeline unavailable because no audio track exists."

        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        if waveform.size == 0:
            return [], [], "Audio timeline unavailable because waveform is empty."

        total_samples = int(waveform.shape[0])
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

        chart_rows: list[dict] = []
        window_rows: list[dict] = []
        for window_no, (start_idx, end_idx) in enumerate(window_ranges):
            self._raise_if_cancelled()
            chunk = waveform[start_idx:end_idx]
            if chunk.size == 0:
                continue
            try:
                predictions = self.hf_audio_pipeline(
                    {"array": chunk.astype(np.float32), "sampling_rate": sample_rate}
                )
            except Exception as error:
                return [], [], f"Audio timeline extraction failed ({error})."

            prediction_list = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
            probs = remap_predictions(prediction_list, AUDIO_LABEL_MAP)
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

        timeline_note = (
            "Audio timeline built with overlapping windows "
            f"({self.config.audio_window_seconds:.1f}s window / {self.config.audio_hop_seconds:.1f}s hop)."
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
        from PIL import Image

        images = [Image.fromarray(frame) for frame in face_frames]
        try:
            predictions = self.hf_video_pipeline(images)
        except Exception as error:
            return ModalitySummary(
                name="video",
                status="missing",
                probabilities={label: 0.0 for label in COMMON_LABELS},
                confidence=0.0,
                quality=0.0,
                note=f"Video model unavailable ({error}).",
            )

        if predictions and not isinstance(predictions[0], list):
            predictions = [predictions]

        aggregated = np.zeros(len(COMMON_LABELS), dtype=np.float64)
        for frame_predictions in predictions:
            remapped = remap_predictions(frame_predictions, VIDEO_LABEL_MAP)
            aggregated += np.array([remapped[label] for label in COMMON_LABELS], dtype=np.float64)
        aggregated = aggregated / aggregated.sum()
        probabilities = {label: float(aggregated[index]) for index, label in enumerate(COMMON_LABELS)}

        quality = min(max(len(face_frames) / float(self.config.sample_frames), 0.30), 1.0)
        return ModalitySummary(
            name="video",
            status="ok",
            probabilities=probabilities,
            confidence=confidence_from_scores(probabilities),
            quality=quality,
            note=(
                "Video model scored face-focused frame crops. "
                f"{frame_note} Face-detected frames: {face_meta['face_detected_frames']}/{len(face_frames)} "
                f"({face_meta['face_ratio'] * 100.0:.1f}%)."
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
            text_modality = self.classify_hf_text(transcript)
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

            from PIL import Image

            images = [Image.fromarray(frame) for frame in face_frames]
            predictions = self.hf_video_pipeline(images)
            if predictions and not isinstance(predictions[0], list):
                predictions = [predictions]

            per_frame_rows: list[dict] = []
            chart_rows: list[dict] = []
            video_frame_rows: list[dict] = []
            video_chart_rows: list[dict] = []
            dominance_counts = {label: 0 for label in COMMON_LABELS}
            available_static_modalities: list[tuple[str, dict[str, float]]] = []
            if text_modality.status == "ok":
                available_static_modalities.append(("text", text_modality.probabilities))
            if audio_modality.status == "ok":
                available_static_modalities.append(("audio", audio_modality.probabilities))

            for i, frame_predictions in enumerate(predictions):
                self._raise_if_cancelled()
                video_probs_raw = remap_predictions(frame_predictions, VIDEO_LABEL_MAP)
                video_probs_for_video_only = self._adjust_video_probabilities(video_probs_raw)
                # Keep Video-Only and Multimodal behavior aligned for video probability adjustments.
                fused_inputs = [("video", video_probs_for_video_only), *available_static_modalities]
                fused_probs, _ = self._weighted_fuse_named(fused_inputs)
                top_emotion = max(fused_probs, key=fused_probs.get)
                dominance_counts[top_emotion] += 1
                row = {
                    "frame_no": int(frame_indices[i]),
                    "video_top_emotion": max(video_probs_raw, key=video_probs_raw.get),
                    "video_top_confidence": float(max(video_probs_raw.values())),
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

    def _build_multimodal_weights(self, modality_names: list[str]) -> dict[str, float]:
        if not modality_names:
            return {}

        unique_names = list(dict.fromkeys(modality_names))
        weights = {name: 1.0 for name in unique_names}
        reduction_ratio = float(np.clip(self.config.text_multimodal_weight_reduction_ratio, 0.0, 1.0))
        other_names = [name for name in unique_names if name != "text"]
        if "text" in weights and other_names and reduction_ratio > 0.0:
            removed_weight = weights["text"] * reduction_ratio
            weights["text"] = weights["text"] - removed_weight
            redistribution = removed_weight / float(len(other_names))
            for name in other_names:
                weights[name] += redistribution

        total = float(sum(weights.values()))
        if total <= 0.0:
            uniform = 1.0 / float(len(unique_names))
            return {name: uniform for name in unique_names}
        return {name: weight / total for name, weight in weights.items()}

    def _weighted_fuse_named(
        self,
        named_probability_sets: list[tuple[str, dict[str, float]]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not named_probability_sets:
            uniform = {label: 1.0 / len(COMMON_LABELS) for label in COMMON_LABELS}
            return uniform, {}

        modality_names = [name for name, _ in named_probability_sets]
        modality_weights = self._build_multimodal_weights(modality_names)
        fused = {label: 0.0 for label in COMMON_LABELS}
        for name, probabilities in named_probability_sets:
            weight = float(modality_weights.get(name, 0.0))
            for label in COMMON_LABELS:
                fused[label] += weight * float(probabilities.get(label, 0.0))

        total = float(sum(fused.values()))
        if total <= 0.0:
            uniform = {label: 1.0 / len(COMMON_LABELS) for label in COMMON_LABELS}
            return uniform, modality_weights
        normalized = {label: score / total for label, score in fused.items()}
        return normalized, modality_weights

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

            modalities: list[ModalitySummary] = [
                self.classify_hf_text(transcript),
                self.classify_hf_video(video_path, workspace),
            ]
            self._raise_if_cancelled()
            modalities.append(self.classify_hf_audio(audio_path))
            self._raise_if_cancelled()

            available_probability_sets: list[tuple[str, dict[str, float]]] = []
            for modality in modalities:
                if modality.status == "ok":
                    available_probability_sets.append((modality.name, modality.probabilities))

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
                "transcript": transcript,
                "transcript_source": transcript_source,
                "fusion_weights": {
                    "text": float(fusion_weights.get("text", 0.0)),
                    "audio": float(fusion_weights.get("audio", 0.0)),
                    "video": float(fusion_weights.get("video", 0.0)),
                },
                "available_modalities": [name for name, _ in available_probability_sets],
                "notes": {
                    "audio": audio_note,
                    "transcript": transcript_note,
                    "fusion": "Text weight is reduced by 30% and redistributed to other available modalities.",
                },
            }
