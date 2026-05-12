from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .manifest import ManifestSample, load_manifest, write_manifest


def create_group_splits(
    manifest_path: str | Path,
    output_dir: str | Path,
    labels: list[str],
    n_splits: int = 5,
    stratified: bool = True,
) -> list[tuple[Path, Path]]:
    samples = load_manifest(manifest_path)
    y = [labels.index(sample.label) for sample in samples]
    groups = [sample.group_id for sample in samples]
    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if stratified
        else GroupKFold(n_splits=n_splits)
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fold_paths: list[tuple[Path, Path]] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(samples, y, groups), start=1):
        train_samples = [samples[index] for index in train_idx]
        val_samples = [samples[index] for index in val_idx]
        train_path = output_root / f"fold_{fold_index}_train.jsonl"
        val_path = output_root / f"fold_{fold_index}_val.jsonl"
        write_manifest(train_samples, train_path)
        write_manifest(val_samples, val_path)
        fold_paths.append((train_path, val_path))

    return fold_paths
