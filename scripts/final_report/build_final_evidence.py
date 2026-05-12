from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


EVIDENCE_FOLDERS = [
    "audio",
    "demo",
    "fusion",
    "repo",
    "report_notes",
    "text",
    "video",
]

INVENTORY_IGNORED_DIRS = {
    ".cache",
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
}

README_TEXT = """# Evidence Folder

Place verified final capstone evidence in this folder only after it has been generated from actual final outputs.

Do not store invented metrics, placeholder predictions, or external model-card scores as project-owned results.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare final report evidence folders and repo inventory.")
    parser.add_argument("--root", default=".", help="Repository root. Defaults to the current directory.")
    parser.add_argument(
        "--evidence-dir",
        default="reports/final_report_evidence",
        help="Evidence directory relative to the repository root.",
    )
    return parser.parse_args()


def git_commit_hash(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "not_available_zip_copy"
    if completed.returncode != 0:
        return "not_available_zip_copy"
    return completed.stdout.strip() or "not_available_zip_copy"


def build_inventory(root: Path, evidence_dir: Path) -> list[str]:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if any(part in INVENTORY_IGNORED_DIRS for part in relative_parts):
            continue
        if path == evidence_dir or evidence_dir in path.parents:
            continue
        relative = path.relative_to(root).as_posix()
        kind = "dir" if path.is_dir() else "file"
        try:
            size = "" if path.is_dir() else str(path.stat().st_size)
        except OSError:
            continue
        rows.append(f"{kind}\t{relative}\t{size}")
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    evidence_dir = root / args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)

    for folder in EVIDENCE_FOLDERS:
        folder_path = evidence_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        if folder in {"text", "audio", "video", "fusion", "demo"}:
            readme_path = folder_path / "README.md"
            if not readme_path.exists():
                readme_path.write_text(README_TEXT, encoding="utf-8")

    repo_dir = evidence_dir / "repo"
    (repo_dir / "git_commit.txt").write_text(git_commit_hash(root) + "\n", encoding="utf-8")
    inventory = build_inventory(root, evidence_dir)
    (repo_dir / "repo_file_inventory.tsv").write_text(
        "type\tpath\tsize_bytes\n" + "\n".join(inventory) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared evidence folders under {evidence_dir}")


if __name__ == "__main__":
    main()
