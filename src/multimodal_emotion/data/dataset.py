from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .manifest import ManifestSample, load_manifest


def load_feature_vector(path: str | None, expected_dim: int) -> tuple[np.ndarray, float]:
    if not path:
        return np.zeros(expected_dim, dtype=np.float32), 0.0

    feature_path = Path(path)
    if not feature_path.exists():
        return np.zeros(expected_dim, dtype=np.float32), 0.0

    feature = np.load(feature_path)
    if feature.ndim > 1:
        feature = feature.mean(axis=0)
    feature = feature.astype(np.float32)

    if feature.shape[-1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch for '{feature_path}'. "
            f"Expected {expected_dim}, found {feature.shape[-1]}."
        )

    return feature, 1.0


class MultimodalFeatureDataset(Dataset):
    def __init__(self, manifest_path: str, label_to_id: dict[str, int], audio_dim: int, video_dim: int) -> None:
        self.samples = load_manifest(manifest_path)
        self.label_to_id = label_to_id
        self.audio_dim = audio_dim
        self.video_dim = video_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample: ManifestSample = self.samples[index]
        audio_features, audio_mask = load_feature_vector(sample.audio_features_path, self.audio_dim)
        video_features, video_mask = load_feature_vector(sample.video_features_path, self.video_dim)
        return {
            "sample_id": sample.sample_id,
            "text": sample.text,
            "label_id": self.label_to_id[sample.label],
            "audio_features": audio_features,
            "video_features": video_features,
            "audio_mask": audio_mask,
            "video_mask": video_mask,
        }


def build_collate_fn(tokenizer, max_text_length: int):
    def collate(batch: list[dict]) -> dict:
        texts = [row["text"] for row in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_tensors="pt",
        )
        return {
            "sample_ids": [row["sample_id"] for row in batch],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "audio_features": torch.tensor(np.stack([row["audio_features"] for row in batch]), dtype=torch.float32),
            "video_features": torch.tensor(np.stack([row["video_features"] for row in batch]), dtype=torch.float32),
            "audio_mask": torch.tensor([row["audio_mask"] for row in batch], dtype=torch.float32),
            "video_mask": torch.tensor([row["video_mask"] for row in batch], dtype=torch.float32),
            "labels": torch.tensor([row["label_id"] for row in batch], dtype=torch.long),
        }

    return collate


def build_dataloader(
    manifest_path: str,
    tokenizer,
    label_to_id: dict[str, int],
    audio_dim: int,
    video_dim: int,
    max_text_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = MultimodalFeatureDataset(manifest_path, label_to_id, audio_dim, video_dim)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collate_fn(tokenizer, max_text_length),
    )
