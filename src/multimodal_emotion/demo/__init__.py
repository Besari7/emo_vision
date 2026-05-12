"""Demo application helpers for upload-based multimodal analysis."""

__all__ = ["MultimodalDemoAnalyzer", "build_demo"]


def __getattr__(name: str):
    if name == "MultimodalDemoAnalyzer":
        from .service import MultimodalDemoAnalyzer

        return MultimodalDemoAnalyzer
    if name == "build_demo":
        from .ui import build_demo

        return build_demo
    raise AttributeError(name)
