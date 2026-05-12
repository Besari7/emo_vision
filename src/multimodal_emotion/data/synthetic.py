from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .manifest import ManifestSample, write_manifest


TEXT_TEMPLATES: dict[str, list[str]] = {
    "neutral": [
        "I am here and everything is fairly normal.",
        "This feels like a regular day so far.",
    ],
    "joy": [
        "I am really happy that this worked out.",
        "This is exciting and I feel great about it.",
    ],
    "sadness": [
        "I feel low and disappointed right now.",
        "This outcome is making me feel sad.",
    ],
    "anger": [
        "I am frustrated and upset about what happened.",
        "This is really making me angry.",
    ],
    "fear": [
        "I am nervous and a little scared about this.",
        "This situation feels risky and frightening.",
    ],
    "disgust": [
        "This feels unpleasant and deeply off-putting.",
        "I really dislike how this turned out.",
    ],
    "surprise": [
        "I did not expect this at all.",
        "That was a surprising change.",
    ],
}


def _label_centroids(labels: list[str], feature_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {label: rng.normal(loc=index * 0.8, scale=0.4, size=feature_dim) for index, label in enumerate(labels)}


def _build_split(
    split_name: str,
    count: int,
    output_dir: Path,
    labels: list[str],
    audio_dim: int,
    video_dim: int,
    rng: np.random.Generator,
) -> list[ManifestSample]:
    audio_dir = output_dir / "features" / "audio"
    video_dir = output_dir / "features" / "video"
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    audio_centroids = _label_centroids(labels, audio_dim, rng)
    video_centroids = _label_centroids(labels, video_dim, rng)

    samples: list[ManifestSample] = []
    label_cycle: Iterable[str] = (labels[index % len(labels)] for index in range(count))

    for index, label in enumerate(label_cycle):
        sample_id = f"{split_name}_{index:04d}"
        text = TEXT_TEMPLATES[label][index % len(TEXT_TEMPLATES[label])]
        audio = audio_centroids[label] + rng.normal(scale=0.35, size=audio_dim)
        video = video_centroids[label] + rng.normal(scale=0.35, size=video_dim)

        audio_path = audio_dir / f"{sample_id}.npy"
        video_path = video_dir / f"{sample_id}.npy"
        np.save(audio_path, audio.astype(np.float32))
        np.save(video_path, video.astype(np.float32))

        samples.append(
            ManifestSample(
                sample_id=sample_id,
                label=label,
                text=text,
                group_id=f"{split_name}_dialogue_{index // 2:04d}",
                speaker_id=f"speaker_{index % 6:02d}",
                audio_features_path=str(audio_path),
                video_features_path=str(video_path),
                metadata={"split": split_name, "transcript_source": "synthetic"},
            )
        )

    return samples


def build_synthetic_dataset(
    output_dir: str | Path,
    labels: list[str],
    audio_dim: int = 64,
    video_dim: int = 64,
    train_size: int = 56,
    val_size: int = 14,
    test_size: int = 14,
    seed: int = 42,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    split_specs = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
    }
    manifest_paths: dict[str, Path] = {}

    for split_name, size in split_specs.items():
        samples = _build_split(split_name, size, root, labels, audio_dim, video_dim, rng)
        manifest_path = root / f"{split_name}.jsonl"
        write_manifest(samples, manifest_path)
        manifest_paths[split_name] = manifest_path

    return manifest_paths
