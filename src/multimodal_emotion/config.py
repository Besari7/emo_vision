from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from multimodal_emotion.labels import CANONICAL_LABELS


@dataclass(slots=True)
class ModelConfig:
    labels: list[str] = field(default_factory=lambda: list(CANONICAL_LABELS))
    text_model_name: str = "roberta-large"
    text_backbone: str = "roberta"
    audio_feature_dim: int = 768
    video_feature_dim: int = 1280
    projection_dim: int = 256
    fusion_hidden_dim: int = 512
    dropout: float = 0.2
    modality_dropout: float = 0.1
    max_text_length: int = 128
    freeze_text_encoder: bool = False
    load_pretrained_text_encoder: bool = True

    @property
    def num_labels(self) -> int:
        return len(self.labels)


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


@dataclass(slots=True)
class ProjectConfig:
    experiment_name: str = "english_multimodal_emotion"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def label_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.model.labels)}


def load_config(path: str | Path) -> ProjectConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model = ModelConfig(**raw.get("model", {}))
    training = TrainingConfig(**raw.get("training", {}))
    return ProjectConfig(
        experiment_name=raw.get("experiment_name", "english_multimodal_emotion"),
        model=model,
        training=training,
    )


def save_config_snapshot(config: ProjectConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
