from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from multimodal_emotion.config import ProjectConfig, load_config
from multimodal_emotion.data.dataset import load_feature_vector
from multimodal_emotion.models import MultimodalEmotionModel
from multimodal_emotion.training.engine import resolve_device


def load_model_for_inference(config_path: str, checkpoint_path: str, device_name: str | None = None):
    config = load_config(config_path)
    device = resolve_device(device_name or config.training.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    model = MultimodalEmotionModel(config.model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return config, tokenizer, model, device


@torch.no_grad()
def predict_single(
    config: ProjectConfig,
    tokenizer,
    model,
    device: torch.device,
    text: str,
    audio_features_path: str | None = None,
    video_features_path: str | None = None,
) -> dict:
    tokenized = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=config.model.max_text_length,
        return_tensors="pt",
    )
    audio_features, audio_mask = load_feature_vector(audio_features_path, config.model.audio_feature_dim)
    video_features, video_mask = load_feature_vector(video_features_path, config.model.video_feature_dim)

    outputs = model(
        input_ids=tokenized["input_ids"].to(device),
        attention_mask=tokenized["attention_mask"].to(device),
        audio_features=torch.tensor(audio_features, dtype=torch.float32, device=device).unsqueeze(0),
        video_features=torch.tensor(video_features, dtype=torch.float32, device=device).unsqueeze(0),
        audio_mask=torch.tensor([audio_mask], dtype=torch.float32, device=device),
        video_mask=torch.tensor([video_mask], dtype=torch.float32, device=device),
    )
    probabilities = outputs["probabilities"].cpu().numpy()[0]
    prediction_index = int(np.argmax(probabilities))
    return {
        "predicted_label": config.model.labels[prediction_index],
        "probabilities": {
            label: float(probability)
            for label, probability in zip(config.model.labels, probabilities, strict=True)
        },
    }
