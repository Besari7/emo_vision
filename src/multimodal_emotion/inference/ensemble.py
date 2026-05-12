from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from multimodal_emotion.config import ModelConfig, ProjectConfig, TrainingConfig, load_config
from multimodal_emotion.data.dataset import load_feature_vector
from multimodal_emotion.data.manifest import load_manifest
from multimodal_emotion.evaluation.metrics import classification_metrics, confusion_records
from multimodal_emotion.models import MultimodalEmotionModel
from multimodal_emotion.training.engine import resolve_device


@dataclass(slots=True)
class EnsembleMember:
    name: str
    weight: float
    config: ProjectConfig
    tokenizer: object
    model: torch.nn.Module


@dataclass(slots=True)
class WeightedEnsemble:
    labels: list[str]
    device: torch.device
    members: list[EnsembleMember]


def _normalized_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(raw_weights.values()))
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return {name: float(value / total) for name, value in raw_weights.items()}


def load_weighted_ensemble(
    members: dict[str, dict[str, str]],
    weights: dict[str, float],
    device_name: str = "auto",
    local_files_only: bool = False,
) -> WeightedEnsemble:
    normalized_weights = _normalized_weights(weights)
    device = resolve_device(device_name)

    bundles: list[EnsembleMember] = []
    labels_reference: list[str] | None = None

    for name, member_info in members.items():
        if name not in normalized_weights:
            continue

        checkpoint_path = member_info["checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_path = member_info.get("config")
        if config_path:
            config = load_config(config_path)
        else:
            raw_config = checkpoint.get("config")
            if not raw_config:
                raise ValueError(f"No config found for ensemble member '{name}'.")
            config = ProjectConfig(
                experiment_name=raw_config.get("experiment_name", "english_multimodal_emotion"),
                model=ModelConfig(**raw_config.get("model", {})),
                training=TrainingConfig(**raw_config.get("training", {})),
            )

        # Checkpoint already includes text encoder weights; prevent redundant remote downloads.
        config.model.load_pretrained_text_encoder = False

        tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name, local_files_only=local_files_only)
        model = MultimodalEmotionModel(config.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        if labels_reference is None:
            labels_reference = list(config.model.labels)
        elif list(config.model.labels) != labels_reference:
            raise ValueError(f"Label mismatch for ensemble member '{name}'.")

        bundles.append(
            EnsembleMember(
                name=name,
                weight=normalized_weights[name],
                config=config,
                tokenizer=tokenizer,
                model=model,
            )
        )

    if not bundles:
        raise ValueError("No valid ensemble members were loaded.")

    assert labels_reference is not None
    return WeightedEnsemble(labels=labels_reference, device=device, members=bundles)


def _model_probabilities(
    member: EnsembleMember,
    device: torch.device,
    text: str,
    audio_features_path: str | None,
    video_features_path: str | None,
) -> np.ndarray:
    tokenized = member.tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=member.config.model.max_text_length,
        return_tensors="pt",
    )
    audio_features, audio_mask = load_feature_vector(audio_features_path, member.config.model.audio_feature_dim)
    video_features, video_mask = load_feature_vector(video_features_path, member.config.model.video_feature_dim)

    with torch.no_grad():
        outputs = member.model(
            input_ids=tokenized["input_ids"].to(device),
            attention_mask=tokenized["attention_mask"].to(device),
            audio_features=torch.tensor(audio_features, dtype=torch.float32, device=device).unsqueeze(0),
            video_features=torch.tensor(video_features, dtype=torch.float32, device=device).unsqueeze(0),
            audio_mask=torch.tensor([audio_mask], dtype=torch.float32, device=device),
            video_mask=torch.tensor([video_mask], dtype=torch.float32, device=device),
        )
    return outputs["probabilities"].cpu().numpy()[0]


def predict_single_ensemble(
    ensemble: WeightedEnsemble,
    text: str,
    audio_features_path: str | None = None,
    video_features_path: str | None = None,
) -> dict:
    weighted_sum = np.zeros(len(ensemble.labels), dtype=np.float64)

    for member in ensemble.members:
        probs = _model_probabilities(
            member=member,
            device=ensemble.device,
            text=text,
            audio_features_path=audio_features_path,
            video_features_path=video_features_path,
        )
        weighted_sum += member.weight * probs

    weighted_sum /= weighted_sum.sum()
    prediction_index = int(np.argmax(weighted_sum))

    return {
        "predicted_label": ensemble.labels[prediction_index],
        "probabilities": {
            label: float(score) for label, score in zip(ensemble.labels, weighted_sum, strict=True)
        },
        "weights": {member.name: member.weight for member in ensemble.members},
    }


def evaluate_ensemble_manifest(ensemble: WeightedEnsemble, manifest_path: str | Path) -> tuple[dict, list[dict]]:
    samples = load_manifest(manifest_path)
    label_to_id = {label: idx for idx, label in enumerate(ensemble.labels)}

    y_true: list[int] = []
    y_pred: list[int] = []
    all_probs: list[np.ndarray] = []
    prediction_rows: list[dict] = []

    for sample in samples:
        result = predict_single_ensemble(
            ensemble=ensemble,
            text=sample.text,
            audio_features_path=sample.audio_features_path,
            video_features_path=sample.video_features_path,
        )
        probs = np.array([result["probabilities"][label] for label in ensemble.labels], dtype=np.float64)
        pred_idx = int(np.argmax(probs))
        true_idx = label_to_id[sample.label]

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        all_probs.append(probs)
        prediction_rows.append(
            {
                "sample_id": sample.sample_id,
                "predicted_label": result["predicted_label"],
                "probabilities": result["probabilities"],
                "weights": result["weights"],
            }
        )

    probs_array = np.asarray(all_probs)
    metrics = classification_metrics(y_true, y_pred, ensemble.labels, probs_array)
    metrics["confusion_matrix"] = confusion_records(y_true, y_pred, ensemble.labels)
    metrics["num_samples"] = len(samples)
    return metrics, prediction_rows
