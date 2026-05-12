from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local multimodal video upload demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload-models", action="store_true", default=True, help="Load models before launching the UI.")
    parser.add_argument(
        "--no-preload-models",
        action="store_false",
        dest="preload_models",
        help="Launch the UI immediately and load models on the first analysis.",
    )
    return parser.parse_args()


def ensure_demo_dependencies(*package_names: str) -> None:
    missing: list[str] = []
    for package_name in package_names:
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        missing_packages = ", ".join(missing)
        raise ImportError(
            "Missing required demo dependencies: "
            f"{missing_packages}. Activate .venv and run `pip install -r requirements-demo.txt`."
        )


def main() -> None:
    args = parse_args()

    ensure_demo_dependencies(
        "torch",
        "transformers",
        "torchvision",
        "gradio",
        "librosa",
        "soundfile",
        "cv2",
        "imageio_ffmpeg",
        "PIL",
        "sklearn",
    )

    from multimodal_emotion.demo.service import MultimodalDemoAnalyzer
    from multimodal_emotion.demo.ui import APP_CSS, build_demo

    analyzer = MultimodalDemoAnalyzer()
    if args.preload_models:
        analyzer.preload_models()

    app = build_demo(analyzer=analyzer)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=APP_CSS,
        allowed_paths=[str(Path(".").resolve())],
    )


if __name__ == "__main__":
    main()
