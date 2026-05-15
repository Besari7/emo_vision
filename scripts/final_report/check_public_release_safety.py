from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable


FORBIDDEN_EXTENSIONS = {
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".h5",
    ".keras",
    ".pkl",
    ".joblib",
    ".npy",
    ".npz",
}

FORBIDDEN_PATH_PREFIXES = [
    "data/external",
    "data/raw",
    "data/processed",
    "datasets",
    "checkpoints",
    "logs",
    "wandb",
    "mlruns",
    "outputs",
]


def get_git_tracked_files(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files


def get_working_tree_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files.append(relative)
    return files


def find_forbidden(files: Iterable[str]) -> tuple[list[str], list[str]]:
    bad_extensions: set[str] = set()
    bad_paths: set[str] = set()
    for rel_path in files:
        normalized = rel_path.replace("\\", "/")
        lower_path = normalized.lower()
        extension = Path(normalized).suffix.lower()
        if extension in FORBIDDEN_EXTENSIONS:
            bad_extensions.add(normalized)
        for prefix in FORBIDDEN_PATH_PREFIXES:
            if lower_path == prefix or lower_path.startswith(f"{prefix}/"):
                bad_paths.add(normalized)
                break
    return sorted(bad_extensions), sorted(bad_paths)


def print_list(title: str, items: list[str]) -> None:
    print(title)
    if not items:
        print("  NONE")
        return
    for item in items:
        print(f"  - {item}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    tracked_files = get_git_tracked_files(root)

    if tracked_files is None:
        print("Public release safety check (git unavailable, scanning working tree).")
        files = get_working_tree_files(root)
    else:
        print("Public release safety check (git-tracked files).")
        files = tracked_files

    bad_extensions, bad_paths = find_forbidden(files)

    print_list("Forbidden extensions detected:", bad_extensions)
    print_list("Forbidden paths detected:", bad_paths)

    if bad_extensions or bad_paths:
        print("FAIL: Remove these files from git tracking or move them to local-only paths.")
        return 1

    print("PASS: No forbidden tracked files detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
