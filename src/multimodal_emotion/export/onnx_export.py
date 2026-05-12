from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _OnnxWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            video_features=video_features,
            audio_mask=audio_mask,
            video_mask=video_mask,
        )
        return outputs["logits"]


def export_model_to_onnx(model, tokenizer, config, output_path: str | Path, opset_version: int = 17) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dummy_tokens = tokenizer(
        ["This is a placeholder sentence for ONNX export."],
        padding=True,
        truncation=True,
        max_length=config.model.max_text_length,
        return_tensors="pt",
    )
    dummy_audio = torch.zeros(1, config.model.audio_feature_dim, dtype=torch.float32)
    dummy_video = torch.zeros(1, config.model.video_feature_dim, dtype=torch.float32)
    dummy_audio_mask = torch.ones(1, dtype=torch.float32)
    dummy_video_mask = torch.ones(1, dtype=torch.float32)

    wrapper = _OnnxWrapper(model).cpu().eval()
    torch.onnx.export(
        wrapper,
        (
            dummy_tokens["input_ids"],
            dummy_tokens["attention_mask"],
            dummy_audio,
            dummy_video,
            dummy_audio_mask,
            dummy_video_mask,
        ),
        output_file,
        input_names=[
            "input_ids",
            "attention_mask",
            "audio_features",
            "video_features",
            "audio_mask",
            "video_mask",
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "audio_features": {0: "batch"},
            "video_features": {0: "batch"},
            "audio_mask": {0: "batch"},
            "video_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
    )
    return output_file
