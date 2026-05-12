from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ManifestSample:
    sample_id: str
    label: str
    text: str
    group_id: str
    speaker_id: str | None = None
    audio_path: str | None = None
    video_path: str | None = None
    audio_features_path: str | None = None
    video_features_path: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict) -> "ManifestSample":
        return cls(
            sample_id=row["sample_id"],
            label=row["label"],
            text=row["text"],
            group_id=row.get("group_id", row["sample_id"]),
            speaker_id=row.get("speaker_id"),
            audio_path=row.get("audio_path"),
            video_path=row.get("video_path"),
            audio_features_path=row.get("audio_features_path"),
            video_features_path=row.get("video_features_path"),
            metadata=row.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "text": self.text,
            "group_id": self.group_id,
            "speaker_id": self.speaker_id,
            "audio_path": self.audio_path,
            "video_path": self.video_path,
            "audio_features_path": self.audio_features_path,
            "video_features_path": self.video_features_path,
            "metadata": self.metadata,
        }


def load_manifest(path: str | Path) -> list[ManifestSample]:
    samples: list[ManifestSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        samples.append(ManifestSample.from_dict(json.loads(line)))
    return samples


def write_manifest(samples: list[ManifestSample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(sample.to_dict(), ensure_ascii=True) for sample in samples)
    target.write_text(f"{payload}\n", encoding="utf-8")


def validate_manifest(samples: list[ManifestSample], labels: list[str]) -> dict:
    if not samples:
        raise ValueError("Manifest is empty.")

    seen_ids: set[str] = set()
    label_counts: dict[str, int] = {label: 0 for label in labels}
    missing_audio_features = 0
    missing_video_features = 0
    groups: set[str] = set()

    for sample in samples:
        if sample.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id found: {sample.sample_id}")
        seen_ids.add(sample.sample_id)

        if sample.label not in label_counts:
            raise ValueError(f"Unknown label '{sample.label}' in manifest.")
        label_counts[sample.label] += 1

        if not sample.text.strip():
            raise ValueError(f"Sample '{sample.sample_id}' has empty text.")

        groups.add(sample.group_id)
        if not sample.audio_features_path:
            missing_audio_features += 1
        if not sample.video_features_path:
            missing_video_features += 1

    return {
        "num_samples": len(samples),
        "num_groups": len(groups),
        "label_counts": label_counts,
        "missing_audio_features": missing_audio_features,
        "missing_video_features": missing_video_features,
    }
