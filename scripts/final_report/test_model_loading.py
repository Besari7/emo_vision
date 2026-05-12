from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_TEXT_MODEL = "artifacts/text_models/roberta_large_goemotions_v2_clean_es"
DEFAULT_AUDIO_MODEL = "artifacts/audio_models/wav2vec2_ravdess_7class"
DEFAULT_VIDEO_MODEL = "artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition"
INSTALL_HINT = "Activate your virtual environment and run: pip install -r requirements-demo.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test local model loading.")
    parser.add_argument("--text", action="store_true", help="Smoke test the local text classifier.")
    parser.add_argument("--audio", action="store_true", help="Smoke test the local audio classifier.")
    parser.add_argument("--video", action="store_true", help="Smoke test the local video classifier.")
    parser.add_argument("--all", action="store_true", help="Run all smoke tests.")
    parser.add_argument("--audio-file", help="Optional WAV/audio file for the audio classifier smoke test.")
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


def test_text() -> bool:
    transformers = require_import("transformers")
    model_path = resolve_model_path(DEFAULT_TEXT_MODEL, "text")
    print(f"Text model path: {model_path}")
    if not ensure_model_dir(model_path, "Text"):
        return False

    classifier = transformers.pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=None,
    )
    predictions = classifier("I am very happy today.")
    print(f"Text smoke test passed: {summarize_predictions(predictions)}")
    return True


def test_video() -> bool:
    transformers = require_import("transformers")
    pil_image = require_import("PIL.Image", "pillow")
    model_path = DEFAULT_VIDEO_MODEL
    print(f"Video model path: {model_path}")
    if not ensure_model_dir(model_path, "Video"):
        return False

    classifier = transformers.pipeline(
        "image-classification",
        model=model_path,
        top_k=None,
    )
    image = pil_image.new("RGB", (224, 224), color=(128, 128, 128))
    predictions = classifier(image)
    print(f"Video smoke test passed: {summarize_predictions(predictions)}")
    return True


def test_audio(audio_file: str | None) -> bool:
    if not audio_file:
        print("Skipping audio test because --audio-file was not provided.")
        return True

    transformers = require_import("transformers")
    librosa = require_import("librosa")
    numpy = require_import("numpy")
    print(
        "Audio lineage: the final audio model, wav2vec2_ravdess_7class, was obtained by further "
        "fine-tuning the CREMA-D fine-tuned Wav2Vec2/SUPERB-ER checkpoint on RAVDESS using the "
        "final seven-class label set."
    )
    model_path = resolve_model_path(DEFAULT_AUDIO_MODEL, "audio")
    if "cremad" in model_path.lower():
        print("WARNING: CREMA-D is a first-stage fine-tuning checkpoint, not the final audio inference path.")
    print(f"Audio model path: {model_path}")
    if not ensure_model_dir(model_path, "Audio"):
        return False

    audio_path = Path(audio_file)
    if not audio_path.is_file():
        print(f"ERROR: Audio file was not found: {audio_path}")
        return False

    waveform, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    classifier = transformers.pipeline(
        "audio-classification",
        model=model_path,
        top_k=None,
    )
    predictions = classifier({"array": numpy.asarray(waveform, dtype=numpy.float32), "sampling_rate": sample_rate})
    print(f"Audio smoke test passed: {summarize_predictions(predictions)}")
    return True


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
        results.append(test_video())
    if run_audio:
        results.append(test_audio(args.audio_file))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
