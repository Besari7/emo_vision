from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check known config files for final canonical label alignment.")
    parser.add_argument("--root", default=".", help="Repository root. Defaults to the current directory.")
    parser.add_argument("--final-config", default="configs/final_capstone.json")
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip local Hugging Face artifact config id2label checks.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_label_lists(value: Any, path: str = "") -> list[tuple[str, list[str]]]:
    found: list[tuple[str, list[str]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in {"labels", "canonical_labels"} and isinstance(child, list) and all(
                isinstance(item, str) for item in child
            ):
                found.append((child_path, list(child)))
            found.extend(extract_label_lists(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(extract_label_lists(child, f"{path}[{index}]"))
    return found


def extract_id2label_labels(config: dict[str, Any]) -> list[str] | None:
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        return None

    indexed_labels: list[tuple[int, str]] = []
    for raw_index, raw_label in id2label.items():
        if not isinstance(raw_label, str):
            return None
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return None
        indexed_labels.append((index, raw_label))

    return [label for _, label in sorted(indexed_labels)]


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    final_config_path = root / args.final_config
    final_config = load_json(final_config_path)
    canonical_labels = list(final_config["canonical_labels"])

    checks: list[tuple[str, str, list[str]]] = []
    checks.append((args.final_config, "canonical_labels", canonical_labels))

    try:
        from multimodal_emotion.config import ProjectConfig
        from multimodal_emotion.labels import normalize_label

        checks.append(("src/multimodal_emotion/config.py", "ProjectConfig().model.labels", ProjectConfig().model.labels))
    except Exception as error:
        print(f"warning: could not inspect src/multimodal_emotion/config.py ({error})")
        normalize_label = lambda label: str(label).strip().lower()  # noqa: E731

    for config_path in sorted((root / "configs").rglob("*.json")):
        data = load_json(config_path)
        for label_path, labels in extract_label_lists(data):
            checks.append((config_path.relative_to(root).as_posix(), label_path, labels))

    mismatches: list[str] = []
    for file_path, label_path, labels in checks:
        status = "OK" if labels == canonical_labels else "MISMATCH"
        print(f"{status}\t{file_path}\t{label_path}\t{labels}")
        if status == "MISMATCH":
            mismatches.append(f"{file_path}:{label_path}")

    artifact_errors: list[str] = []
    artifact_warnings: list[str] = []
    if not args.skip_artifacts:
        for artifact_config_path in sorted((root / "artifacts").rglob("config.json")):
            try:
                data = load_json(artifact_config_path)
            except OSError as error:
                print(f"warning: could not read {artifact_config_path} ({error})")
                continue

            artifact_labels = extract_id2label_labels(data)
            if artifact_labels is None:
                continue

            artifact_labels = [normalize_label(label) for label in artifact_labels]
            relative_path = artifact_config_path.relative_to(root).as_posix()
            if sorted(artifact_labels) != sorted(canonical_labels):
                status = "ARTIFACT_ERROR"
                artifact_errors.append(relative_path)
            elif artifact_labels != canonical_labels:
                status = "ARTIFACT_CONVERT"
                artifact_warnings.append(relative_path)
            else:
                status = "ARTIFACT_OK"
            print(f"{status}\t{relative_path}\tid2label\t{artifact_labels}")

    if mismatches:
        print("\nLabel order mismatches require conversion before final fusion/reporting:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        raise SystemExit(1)

    if artifact_warnings:
        print("\nArtifact label orders differ from the canonical order and must be converted by label name before reporting:")
        for warning in artifact_warnings:
            print(f"- {warning}")

    if artifact_errors:
        print("\nArtifact label sets do not match the final canonical label set:")
        for error in artifact_errors:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
