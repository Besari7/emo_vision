"""Inference helpers."""

from .audio import AudioEmotionPredictor, predict_audio
from .fusion import FusionEngine, fuse_predictions
from .result import PredictionResult
from .runtime_config import ConfidenceGatingConfig, InferenceRuntimeConfig, load_runtime_config, resolve_model_path
from .text import TextEmotionPredictor, predict_text, predict_text_label
from .video import VideoEmotionPredictor, predict_video


def load_model_for_inference(*args, **kwargs):
    from .predictor import load_model_for_inference as _load_model_for_inference

    return _load_model_for_inference(*args, **kwargs)


def predict_single(*args, **kwargs):
    from .predictor import predict_single as _predict_single

    return _predict_single(*args, **kwargs)


def load_weighted_ensemble(*args, **kwargs):
    from .ensemble import load_weighted_ensemble as _load_weighted_ensemble

    return _load_weighted_ensemble(*args, **kwargs)


def predict_single_ensemble(*args, **kwargs):
    from .ensemble import predict_single_ensemble as _predict_single_ensemble

    return _predict_single_ensemble(*args, **kwargs)


def evaluate_ensemble_manifest(*args, **kwargs):
    from .ensemble import evaluate_ensemble_manifest as _evaluate_ensemble_manifest

    return _evaluate_ensemble_manifest(*args, **kwargs)

__all__ = [
    "load_model_for_inference",
    "predict_single",
    "load_weighted_ensemble",
    "predict_single_ensemble",
    "evaluate_ensemble_manifest",
    "AudioEmotionPredictor",
    "predict_audio",
    "FusionEngine",
    "fuse_predictions",
    "PredictionResult",
    "ConfidenceGatingConfig",
    "InferenceRuntimeConfig",
    "load_runtime_config",
    "resolve_model_path",
    "TextEmotionPredictor",
    "predict_text",
    "predict_text_label",
    "VideoEmotionPredictor",
    "predict_video",
]
