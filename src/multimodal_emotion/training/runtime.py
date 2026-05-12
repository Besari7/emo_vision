from __future__ import annotations


def ensure_training_dependencies(*extra_packages: str) -> None:
    missing: list[str] = []
    for package_name in ("torch", "transformers", "torchvision", *extra_packages):
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        missing_packages = ", ".join(missing)
        raise ImportError(
            "Missing required training dependencies: "
            f"{missing_packages}. Install them with `pip install -r requirements.txt`."
            " For the local demo environment, use `pip install -r requirements-demo.txt`."
        )
