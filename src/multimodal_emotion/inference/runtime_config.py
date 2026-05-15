from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "final_capstone.json"

MODEL_ENV_VARS = {
    "text": "EMOVISION_TEXT_MODEL",
    "audio": "EMOVISION_AUDIO_MODEL",
    "video": "EMOVISION_VIDEO_MODEL",
}

TEMPERATURE_ENV_VARS = {
    "text": "EMOVISION_TEXT_TEMPERATURE",
    "audio": "EMOVISION_AUDIO_TEMPERATURE",
    "video": "EMOVISION_VIDEO_TEMPERATURE",
}

DEFAULT_MODEL_PATHS = {
    "text": PROJECT_ROOT / "artifacts" / "text_models" / "roberta_large_goemotions_ekman_v2_continued_from_direct7",
    "audio": PROJECT_ROOT / "artifacts" / "audio_models" / "wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop",
    "video": PROJECT_ROOT / "artifacts" / "video_models" / "vit_based_fer_model",
}

TEMPERATURES = {
    "text": 1.0,
    "audio": 1.4,
    "video": 1.3,
}

GLOBAL_WEIGHTS = {
    "text": 0.55,
    "audio": 0.20,
    "video": 0.25,
}

CONFIDENCE_GATING = {
    "enabled": False,
    "video_min_conf_zero": 0.25,
    "video_min_conf_low": 0.35,
    "video_low_multiplier": 0.25,
}


class ModelPathConfigError(RuntimeError):
    pass


@dataclass(slots=True)
class RuntimeModelPaths:
    text: str
    audio: str
    video: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "audio": self.audio, "video": self.video}


@dataclass(slots=True)
class ConfidenceGatingConfig:
    enabled: bool = False
    video_min_conf_zero: float = 0.25
    video_min_conf_low: float = 0.35
    video_low_multiplier: float = 0.25

    def as_dict(self) -> dict[str, bool | float]:
        return {
            "enabled": self.enabled,
            "video_min_conf_zero": self.video_min_conf_zero,
            "video_min_conf_low": self.video_min_conf_low,
            "video_low_multiplier": self.video_low_multiplier,
        }


@dataclass(slots=True)
class InferenceRuntimeConfig:
    model_paths: RuntimeModelPaths
    temperatures: dict[str, float] = field(default_factory=lambda: dict(TEMPERATURES))
    global_weights: dict[str, float] = field(default_factory=lambda: dict(GLOBAL_WEIGHTS))
    confidence_gating: ConfidenceGatingConfig = field(default_factory=ConfidenceGatingConfig)


def _has_model_files(path: Path) -> bool:
    return (path / "config.json").is_file() and (
        (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    )


def _is_probable_hf_model_ref(value: str) -> bool:
    return "/" in value and "\\" not in value and not value.startswith(".") and not Path(value).is_absolute()


def _read_final_config(config_path: str | Path | None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _configured_path_from_file(modality: str, config: dict) -> str | None:
    if modality == "text":
        branch = config.get("text_branch", {})
    elif modality == "audio":
        branch = config.get("audio_branch", {})
    elif modality == "video":
        branch = config.get("visual_branch", {})
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    for key in ("final_artifact_path", "model_path", "artifact_path"):
        value = branch.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _source_model_path(modality: str, config: dict) -> tuple[str, str]:
    env_var = MODEL_ENV_VARS[modality]
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value, f"environment variable {env_var}"

    config_value = _configured_path_from_file(modality, config)
    if config_value:
        return config_value, f"config file {DEFAULT_CONFIG_PATH.relative_to(PROJECT_ROOT)}"

    return str(DEFAULT_MODEL_PATHS[modality]), "project default"


def _configured_float_map(config: dict, key: str, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    candidates = [
        config.get(key),
        config.get("inference", {}).get(key) if isinstance(config.get("inference"), dict) else None,
        config.get("fusion", {}).get(key) if isinstance(config.get("fusion"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for modality in values:
            raw_value = candidate.get(modality)
            if raw_value is not None:
                values[modality] = float(raw_value)
    return values


def _temperature_config(config: dict) -> dict[str, float]:
    values = _configured_float_map(config, "temperatures", TEMPERATURES)
    for modality, env_var in TEMPERATURE_ENV_VARS.items():
        raw_value = os.environ.get(env_var)
        if raw_value is not None and raw_value.strip():
            values[modality] = float(raw_value)
    return values


def _global_weight_config(config: dict) -> dict[str, float]:
    return _configured_float_map(config, "global_weights", GLOBAL_WEIGHTS)


def _configured_confidence_gating(config: dict) -> ConfidenceGatingConfig:
    values = dict(CONFIDENCE_GATING)
    candidates = [
        config.get("confidence_gating"),
        config.get("inference", {}).get("confidence_gating") if isinstance(config.get("inference"), dict) else None,
        config.get("fusion", {}).get("confidence_gating") if isinstance(config.get("fusion"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if "enabled" in candidate:
            values["enabled"] = bool(candidate["enabled"])
        for key in ("video_min_conf_zero", "video_min_conf_low", "video_low_multiplier"):
            if candidate.get(key) is not None:
                values[key] = float(candidate[key])
    return ConfidenceGatingConfig(
        enabled=bool(values["enabled"]),
        video_min_conf_zero=float(values["video_min_conf_zero"]),
        video_min_conf_low=float(values["video_min_conf_low"]),
        video_low_multiplier=float(values["video_low_multiplier"]),
    )


def resolve_model_path(
    modality: str,
    configured_path: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    validate: bool = True,
) -> str:
    if modality not in MODEL_ENV_VARS:
        raise ValueError(f"Unknown modality: {modality!r}")

    config = _read_final_config(config_path)
    source = "explicit argument"
    raw_path: str
    if configured_path is None:
        raw_path, source = _source_model_path(modality, config)
    else:
        raw_path = str(configured_path)

    path = Path(raw_path).expanduser()
    if not path.is_absolute() and not _is_probable_hf_model_ref(raw_path):
        path = PROJECT_ROOT / path

    if _is_probable_hf_model_ref(raw_path) and not path.exists():
        return raw_path

    attempted = path
    nested_path = path / "best_model"
    checked_nested = False

    if path.is_dir():
        if _has_model_files(path):
            return str(path)
        checked_nested = True
        if _has_model_files(nested_path):
            return str(nested_path)
        if not validate:
            return str(path)
    elif not validate:
        return str(path)

    env_var = MODEL_ENV_VARS[modality]
    if validate:
        nested_note = f" Checked nested fallback: {nested_path}." if checked_nested or nested_path else ""
        raise ModelPathConfigError(
            f"{modality} model path could not be resolved from {source}. "
            f"Attempted path: {attempted}.{nested_note} "
            f"Set {env_var} to the model directory or to its best_model subdirectory."
        )

    return str(path)


def load_runtime_config(
    *,
    config_path: str | Path | None = None,
    validate_paths: bool = True,
) -> InferenceRuntimeConfig:
    config = _read_final_config(config_path)
    paths = {
        modality: resolve_model_path(modality, config_path=config_path, validate=validate_paths)
        for modality in ("text", "audio", "video")
    }
    return InferenceRuntimeConfig(
        model_paths=RuntimeModelPaths(**paths),
        temperatures=_temperature_config(config),
        global_weights=_global_weight_config(config),
        confidence_gating=_configured_confidence_gating(config),
    )
