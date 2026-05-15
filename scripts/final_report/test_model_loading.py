from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_TEXT_MODEL = "artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7"
DEFAULT_AUDIO_MODEL = "artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop"
DEFAULT_VIDEO_MODEL = "artifacts/video_models/vit_based_fer_model"
INSTALL_HINT = "Activate your virtual environment and run: pip install -r requirements-demo.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test local model loading.")
    parser.add_argument("--text", action="store_true", help="Smoke test the local text classifier.")
    parser.add_argument("--audio", action="store_true", help="Smoke test the local audio classifier.")
    parser.add_argument("--video", action="store_true", help="Smoke test the local video classifier.")
    parser.add_argument("--all", action="store_true", help="Run all smoke tests.")
    parser.add_argument("--audio-file", help="Optional WAV/audio file for the audio classifier smoke test.")
    parser.add_argument("--video-file", help="Optional video file for the video classifier smoke test.")
    return parser.parse_args()


def require_import(module_name: str, package_name: str | None = None) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        display_name = package_name or module_name
        print(f"Missing dependency: {display_name}. {INSTALL_HINT}")
        raise SystemExit(1) from error


def ensure_model_dir(path: str, label: str) -> bool:
    model_path = Path(path)
    if model_path.is_dir():
        return True
    print(f"ERROR: {label} model directory was not found: {model_path}")
    return False


def has_model_files(path: Path) -> bool:
    has_config = (path / "config.json").is_file()
    has_weights = (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    return has_config and has_weights


def resolve_model_path(path: str, label: str) -> str:
    model_path = Path(path)
    nested_path = model_path / "best_model"
    if model_path.is_dir() and not has_model_files(model_path) and has_model_files(nested_path):
        print(f"Using nested {label} model files: {nested_path}")
        return str(nested_path)
    return path


def summarize_predictions(predictions: Any) -> str:
    rows = predictions
    if isinstance(rows, list) and rows and isinstance(rows[0], list):
        rows = rows[0]
    if not isinstance(rows, list) or not rows:
        return "no predictions returned"
    top = max(rows, key=lambda item: float(item.get("score", 0.0)))
    return f"top label={top.get('label')} score={float(top.get('score', 0.0)):.4f}"


def ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def assert_prediction_result(result: Any, label: str) -> bool:
    from multimodal_emotion.labels import CANONICAL_LABELS

    if not result.available:
        print(f"ERROR: {label} predictor returned unavailable: {result.error}")
        return False
    if result.labels != CANONICAL_LABELS:
        print(f"ERROR: {label} labels are not canonical: {result.labels}")
        return False
    if result.probs is None or len(result.probs) != len(CANONICAL_LABELS):
        print(f"ERROR: {label} did not return 7 probabilities.")
        return False
    prob_sum = float(sum(result.probs))
    if abs(prob_sum - 1.0) > 1e-4:
        print(f"ERROR: {label} probabilities sum to {prob_sum:.6f}, not 1.")
        return False
    print(f"{label} smoke test passed: top label={result.pred_label} score={result.confidence:.4f}")
    return True


def print_video_result_details(result: Any) -> None:
    probs = result.probs or []
    prob_sum = float(sum(probs)) if probs else float("nan")
    print("Video result details:")
    print(f"  available={result.available}")
    print(f"  pred_label={result.pred_label}")
    print(f"  confidence={float(result.confidence):.6f}")
    print(f"  labels={result.labels}")
    print(f"  len(probs)={len(probs)}")
    print(f"  sum(probs)={prob_sum:.6f}")
    print(f"  quality={result.quality}")


def test_text() -> bool:
    ensure_src_on_path()
    require_import("transformers")
    from multimodal_emotion.inference.text import TextEmotionPredictor

    model_path = resolve_model_path(DEFAULT_TEXT_MODEL, "text")
    print(f"Text model path: {model_path}")
    if not ensure_model_dir(model_path, "Text"):
        return False

    predictor = TextEmotionPredictor(model_path=model_path)
    return assert_prediction_result(predictor.predict("I am very happy today."), "Text")


def test_video(video_file: str | None = None) -> bool:
    ensure_src_on_path()
    require_import("transformers")
    pil_image = require_import("PIL.Image", "pillow")
    from multimodal_emotion.inference.video import VideoEmotionPredictor

    model_path = DEFAULT_VIDEO_MODEL
    print(f"Video model path: {model_path}")
    if not ensure_model_dir(model_path, "Video"):
        return False

    predictor = VideoEmotionPredictor(model_path=model_path)
    if video_file:
        require_import("cv2", "opencv-python")
        video_path = Path(video_file)
        if not video_path.is_file():
            print(f"ERROR: Video file was not found: {video_path}")
            return False
        result = predictor.predict(video_path)
    else:
        image = pil_image.new("RGB", (224, 224), color=(128, 128, 128))
        result = predictor.predict_images([image], fallback=True, quality={"quality_weight_multiplier": 0.5})

    print_video_result_details(result)
    return assert_prediction_result(result, "Video")



def test_audio(audio_file: str | None) -> bool:
    ensure_src_on_path()
    require_import("transformers")
    librosa = require_import("librosa")
    numpy = require_import("numpy")
    from multimodal_emotion.inference.audio import AudioEmotionPredictor

    print(
        "Audio lineage: the final audio inference artifact is "
        "wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop, a seven-class Wav2Vec2/XLS-R "
        "classifier aligned to the canonical label set."
    )
    model_path = resolve_model_path(DEFAULT_AUDIO_MODEL, "audio")
    if "ravdess_7class" in model_path.lower() and "xlsr" not in model_path.lower():
        print("WARNING: This looks like the older RAVDESS audio artifact, not the current XLS-R audio path.")
    print(f"Audio model path: {model_path}")
    if not ensure_model_dir(model_path, "Audio"):
        return False

    predictor = AudioEmotionPredictor(model_path=model_path)
    if audio_file:
        audio_path = Path(audio_file)
        if not audio_path.is_file():
            print(f"ERROR: Audio file was not found: {audio_path}")
            return False
        waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    else:
        sample_rate = 16000
        t = numpy.linspace(0.0, 1.0, sample_rate, endpoint=False)
        waveform = 0.05 * numpy.sin(2.0 * numpy.pi * 220.0 * t)
    return assert_prediction_result(
        predictor.predict_waveform(numpy.asarray(waveform, dtype=numpy.float32), sample_rate),
        "Audio",
    )


def main() -> int:
    args = parse_args()
    print(f"Python executable: {sys.executable}")
    print("This is a smoke test, not final evaluation.")

    run_text = args.text or args.all
    run_audio = args.audio or args.all
    run_video = args.video or args.all
    if not (run_text or run_audio or run_video):
        print("No smoke test selected. Use --text, --audio, --video, or --all.")
        return 2

    results: list[bool] = []
    if run_text:
        results.append(test_text())
    if run_video:
        results.append(test_video(args.video_file))
    if run_audio:
        results.append(test_audio(args.audio_file))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
