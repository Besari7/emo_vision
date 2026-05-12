from __future__ import annotations

from pathlib import Path


MODEL_PATHS = {
    "text": Path("artifacts/text_models/roberta_large_goemotions_v2_clean_es"),
    "audio": Path("artifacts/audio_models/wav2vec2_ravdess_7class"),
    "video": Path("artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition"),
}

COMMON_FILES = {
    "text": [
        ("config", ["config.json"]),
        ("weights", ["model.safetensors", "pytorch_model.bin"]),
        ("tokenizer", ["tokenizer.json", "tokenizer_config.json"]),
    ],
    "audio": [
        ("config", ["config.json"]),
        ("weights", ["model.safetensors", "pytorch_model.bin"]),
        ("preprocessor", ["preprocessor_config.json", "feature_extractor_config.json"]),
    ],
    "video": [
        ("config", ["config.json"]),
        ("weights", ["model.safetensors", "pytorch_model.bin"]),
        ("preprocessor", ["preprocessor_config.json", "image_processor_config.json"]),
    ],
}


def has_any(directory: Path, candidates: list[str]) -> bool:
    return any((directory / candidate).exists() for candidate in candidates)


def print_common_file_status(name: str, directory: Path, model_type: str) -> None:
    if not directory.is_dir():
        return

    for label, candidates in COMMON_FILES[model_type]:
        status = "FOUND" if has_any(directory, candidates) else "MISSING"
        joined = " or ".join(candidates)
        print(f"  {status}: {label} ({joined})")


def check_path(name: str, path: Path, model_type: str | None = None) -> bool:
    found = path.is_dir()
    print(f"{'FOUND' if found else 'MISSING'}: {name} -> {path}")
    if found and model_type is not None:
        print_common_file_status(name, path, model_type)
    return found


def has_model_files(directory: Path, model_type: str) -> bool:
    if not directory.is_dir():
        return False
    return all(has_any(directory, candidates) for _, candidates in COMMON_FILES[model_type])


def check_model_package(name: str, package_path: Path, model_type: str) -> bool:
    package_found = check_path(f"{name}_package", package_path)
    if not package_found:
        return False

    if has_model_files(package_path, model_type):
        print_common_file_status(name, package_path, model_type)
        return True

    nested_path = package_path / "best_model"
    return check_path(f"{name}_model_files", nested_path, model_type) and has_model_files(nested_path, model_type)


def main() -> int:
    print("Checking local model artifact directories. Models are not loaded.")
    print(
        "Audio lineage: the final audio model, wav2vec2_ravdess_7class, was obtained by further "
        "fine-tuning the CREMA-D fine-tuned Wav2Vec2/SUPERB-ER checkpoint on RAVDESS using the "
        "final seven-class label set."
    )

    text_found = check_model_package("text", MODEL_PATHS["text"], "text")
    audio_found = check_model_package("audio", MODEL_PATHS["audio"], "audio")
    video_found = check_model_package("video", MODEL_PATHS["video"], "video")

    exit_code = 0
    if not text_found:
        print(f"ERROR: Required text model files are missing under: {MODEL_PATHS['text']}")
        exit_code = 1
    if not video_found:
        print(f"ERROR: Required video model files are missing under: {MODEL_PATHS['video']}")
        exit_code = 1
    if not audio_found:
        print(f"ERROR: Required audio model files are missing under: {MODEL_PATHS['audio']}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
