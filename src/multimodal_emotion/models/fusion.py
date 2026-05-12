from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel, RobertaConfig, RobertaModel

from multimodal_emotion.config import ModelConfig


class ProjectionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultimodalEmotionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = self._build_text_encoder(config)
        text_hidden_size = self.text_encoder.config.hidden_size

        if config.freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        self.text_projection = ProjectionBlock(text_hidden_size, config.projection_dim, config.dropout)
        self.audio_projection = ProjectionBlock(config.audio_feature_dim, config.projection_dim, config.dropout)
        self.video_projection = ProjectionBlock(config.video_feature_dim, config.projection_dim, config.dropout)

        self.gating = nn.Sequential(
            nn.Linear(config.projection_dim * 3 + 3, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.projection_dim + 3, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_labels),
        )

    def _build_text_encoder(self, config: ModelConfig):
        if config.load_pretrained_text_encoder:
            return AutoModel.from_pretrained(config.text_model_name)

        if "roberta" in config.text_model_name.lower():
            return RobertaModel(
                RobertaConfig(
                    vocab_size=50265,
                    max_position_embeddings=514,
                    type_vocab_size=1,
                    pad_token_id=1,
                    bos_token_id=0,
                    eos_token_id=2,
                )
            )
        if "bert" in config.text_model_name.lower():
            return BertModel(BertConfig())

        raise ValueError(
            "Only BERT/Roberta random initialization is supported without pretrained weights. "
            f"Received '{config.text_model_name}'."
        )

    def _apply_modality_dropout(
        self,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.config.modality_dropout <= 0:
            return audio_mask, video_mask

        drop_audio = torch.bernoulli(
            torch.full_like(audio_mask, self.config.modality_dropout)
        )
        drop_video = torch.bernoulli(
            torch.full_like(video_mask, self.config.modality_dropout)
        )
        audio_mask = audio_mask * (1.0 - drop_audio)
        video_mask = video_mask * (1.0 - drop_video)
        return audio_mask, video_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_hidden = text_outputs.pooler_output
        else:
            text_hidden = text_outputs.last_hidden_state[:, 0]

        audio_mask, video_mask = self._apply_modality_dropout(audio_mask, video_mask)

        text_embedding = self.text_projection(text_hidden)
        audio_embedding = self.audio_projection(audio_features) * audio_mask.unsqueeze(-1)
        video_embedding = self.video_projection(video_features) * video_mask.unsqueeze(-1)

        availability = torch.stack(
            [
                torch.ones_like(audio_mask),
                audio_mask,
                video_mask,
            ],
            dim=1,
        )
        embeddings = torch.stack([text_embedding, audio_embedding, video_embedding], dim=1)

        gate_input = torch.cat(
            [
                text_embedding,
                audio_embedding,
                video_embedding,
                availability,
            ],
            dim=1,
        )
        gate_logits = self.gating(gate_input)
        gate_logits = gate_logits.masked_fill(availability == 0, -1e4)
        weights = torch.softmax(gate_logits, dim=1)

        fused = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        classifier_input = torch.cat([fused, availability], dim=1)
        logits = self.classifier(classifier_input)
        probabilities = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "weights": weights,
        }
