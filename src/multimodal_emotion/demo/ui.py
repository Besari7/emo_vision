from __future__ import annotations

import base64
from collections import Counter
from pathlib import Path

import pandas as pd
import gradio as gr

from .service import COMMON_LABELS, MultimodalDemoAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEADER_BANNER_PATH = PROJECT_ROOT / "assets" / "ui" / "header_emovision.png"


def _load_header_banner_data_uri() -> str:
    if not HEADER_BANNER_PATH.exists():
        return ""
    encoded = base64.b64encode(HEADER_BANNER_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


HEADER_BANNER_DATA_URI = _load_header_banner_data_uri()


APP_CSS = """
:root {
  --ink: #eaf0ff;
  --panel: #0f172a;
  --panel-soft: #1e293b;
  --line: #334155;
  --accent: #3b82f6;
}

.gradio-container {
  background: #000a1f;
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}

.app-shell {
  max-width: 1280px;
  margin: 0 auto;
}

.hero-banner-native {
  width: 100%;
  margin: 0;
  border-radius: 20px;
  overflow: hidden;
}

.hero-banner-native img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 0;
  border: 0;
  box-shadow: none;
  pointer-events: none;
  user-select: none;
}

.hero-banner-fallback {
  margin: 0 0 14px 0;
  padding: 0;
}

.hero-banner-host {
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

@media (max-width: 900px) {
  .hero-banner-native img {
    border-radius: 16px;
  }
}

.hero-card, .result-card {
  border: 1px solid var(--line);
  background: rgba(15, 23, 42, 0.92);
  border-radius: 18px;
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.34);
}

.hero-card {
  padding: 20px 22px 18px;
  margin-bottom: 16px;
  background:
    radial-gradient(circle at 14% 18%, rgba(56, 189, 248, 0.13), transparent 42%),
    radial-gradient(circle at 86% 78%, rgba(59, 130, 246, 0.15), transparent 48%),
    linear-gradient(145deg, rgba(10, 20, 40, 0.96), rgba(16, 33, 63, 0.92));
  border: 1px solid rgba(96, 165, 250, 0.3);
  box-shadow:
    inset 0 0 0 1px rgba(37, 99, 235, 0.16),
    0 14px 36px rgba(2, 6, 23, 0.5);
}

.result-card {
  padding: 14px 16px;
}

.hero-card h1 {
  font-size: 2.18rem;
  margin: 0;
  color: #eef5ff;
  letter-spacing: 0.015em;
  line-height: 1.05;
}

.hero-card p {
  color: #bed0ec;
  line-height: 1.45;
  margin: 9px 0 0;
}

.hero-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
}

.hero-copy {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.hero-subtitle {
  font-size: 1.02rem;
  font-weight: 650;
  color: #8ec5ff;
  letter-spacing: 0.02em;
}

.hero-description {
  font-size: 0.97rem;
  max-width: 720px;
  color: #c7d8f5;
}

.hero-project-name {
  font-size: 0.86rem;
  color: #72b9ff;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 1px;
}

.hero-badge {
  align-self: center;
  white-space: nowrap;
  font-size: 0.84rem;
  font-weight: 700;
  letter-spacing: 0.035em;
  color: #d6e9ff;
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.34), rgba(30, 64, 175, 0.18));
  border: 1px solid rgba(96, 165, 250, 0.42);
  border-radius: 999px;
  padding: 9px 14px;
  box-shadow:
    inset 0 0 0 1px rgba(191, 219, 254, 0.12),
    0 8px 22px rgba(30, 64, 175, 0.28);
}

.hero-divider {
  margin-top: 14px;
  position: relative;
  height: 1px;
  background: linear-gradient(90deg, rgba(96, 165, 250, 0.36), rgba(96, 165, 250, 0.08) 76%, rgba(96, 165, 250, 0.28));
}

.hero-divider::after {
  content: "";
  position: absolute;
  right: 0;
  top: -3px;
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #5ab5ff;
  box-shadow: 0 0 14px rgba(90, 181, 255, 0.9);
}

@media (max-width: 820px) {
  .hero-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .hero-badge {
    align-self: flex-start;
  }

  .hero-card h1 {
    font-size: 1.9rem;
  }
}

.blue-button {
  background: linear-gradient(120deg, #2563eb, #3b82f6 50%, #60a5fa);
  color: #f8fbff;
  border: none;
}

.spotlight-card {
  border: 1px solid var(--line);
  background: linear-gradient(155deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
  border-radius: 18px;
  padding: 16px 18px;
}

.spotlight-title {
  font-size: 1.05rem;
  color: #93c5fd;
  margin-bottom: 6px;
}

.spotlight-primary {
  font-size: 2.2rem;
  font-weight: 800;
  color: #f8fbff;
  line-height: 1.1;
}

.spotlight-primary-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.spotlight-jump-btn {
  border: 1px solid rgba(96, 165, 250, 0.45);
  background: linear-gradient(130deg, rgba(37, 99, 235, 0.35), rgba(59, 130, 246, 0.2));
  color: #dbeafe;
  border-radius: 999px;
  padding: 7px 12px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  cursor: pointer;
  white-space: nowrap;
}

.spotlight-jump-btn:hover {
  filter: brightness(1.08);
}

.spotlight-sub {
  font-size: 0.95rem;
  color: #bfdbfe;
  margin-bottom: 10px;
}

.spotlight-top3 {
  margin: 0;
  padding-left: 1rem;
  color: #dbeafe;
}

.spotlight-top3 li {
  margin: 4px 0;
}

.spotlight-divider {
  border-top: 1px solid rgba(148,163,184,0.35);
  margin: 12px 0 10px;
}

.spotlight-small {
  font-size: 0.9rem;
  color: #cbd5e1;
  margin: 3px 0;
}

.spotlight-foot {
  font-size: 0.82rem;
  color: #94a3b8;
  margin-top: 8px;
}

@media (max-width: 820px) {
  .spotlight-primary-row {
    flex-direction: column;
    gap: 8px;
  }
}
"""


def _build_spotlight_html(result: dict) -> str:
    stats_rows = list(result.get("stats_rows", []))
    modality_rows = list(result.get("modality_rows", []))
    video_frame_rows = list(result.get("video_frame_rows", []))

    if not stats_rows:
        return "<section class='spotlight-card'><div class='spotlight-title'>Detected Overall Emotion</div><div class='spotlight-sub'>Upload a video to begin.</div></section>"

    top3 = sorted(stats_rows, key=lambda row: float(row.get("mean_probability", 0.0)), reverse=True)[:3]
    primary = top3[0]
    primary_emotion = str(primary["emotion"])
    primary_score = float(primary["mean_probability"]) * 100.0

    top3_items = "".join(
        f"<li><strong>{row['emotion']}</strong>: {float(row['mean_probability']) * 100.0:.1f}%</li>"
        for row in top3
    )

    video_counter = Counter(row.get("top_emotion", "unavailable") for row in video_frame_rows)
    video_dominant = video_counter.most_common(1)[0][0].title() if video_counter else "Unavailable"

    def modality_dominant(name: str) -> str:
        for row in modality_rows:
            if str(row.get("modality", "")).lower() == name:
                return str(row.get("top_emotion", "Unavailable")).title()
        return "Unavailable"

    audio_dominant = modality_dominant("audio")
    text_dominant = modality_dominant("text")
    return f"""
    <section class="spotlight-card">
      <div class="spotlight-title">Detected Overall Emotion</div>
      <div class="spotlight-primary-row">
        <div class="spotlight-primary">{primary_emotion}</div>
        <button
          class="spotlight-jump-btn"
          type="button"
          onclick="document.getElementById('statistics-section')?.scrollIntoView({{behavior:'smooth', block:'start'}});"
        >
          Go to statistics
        </button>
      </div>
      <div class="spotlight-sub">Most dominant multimodal emotion: {primary_score:.1f}%</div>
      <ol class="spotlight-top3">{top3_items}</ol>
      <div class="spotlight-divider"></div>
      <div class="spotlight-small">Video dominant emotion: <strong>{video_dominant}</strong></div>
      <div class="spotlight-small">Audio dominant emotion: <strong>{audio_dominant}</strong></div>
      <div class="spotlight-small">Text dominant emotion: <strong>{text_dominant}</strong></div>
    </section>
    """


def build_demo(analyzer: MultimodalDemoAnalyzer | None = None) -> gr.Blocks:
    analyzer = analyzer or MultimodalDemoAnalyzer()

    def run_frame_statistics(
        video_path: str | None,
        max_frames: int,
        transcript_override: str,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
      pd.DataFrame,
        str,
        str,
        str | None,
        dict,
    ]:
        if not video_path:
            raise gr.Error("Upload a video file before running frame-level emotion analysis.")

        analyzer.clear_cancel()
        result = analyzer.analyze_video_frame_statistics(
            video_path,
            max_frames=max_frames,
            transcript_override=transcript_override or None,
        )

        fused_chart_df = pd.DataFrame(result["chart_rows"])
        fused_frame_df = pd.DataFrame(result["per_frame_rows"])
        video_chart_df = pd.DataFrame(result["video_chart_rows"])
        video_frame_df = pd.DataFrame(result["video_frame_rows"])
        audio_chart_df = pd.DataFrame(result["audio_chart_rows"])
        audio_window_df = pd.DataFrame(result["audio_window_rows"])
        stats_df = pd.DataFrame(result["stats_rows"])
        modality_df = pd.DataFrame(result["modality_rows"])
        text_df = pd.DataFrame(result["text_prob_rows"])
        text_chunk_df = pd.DataFrame(result.get("text_chunk_rows", []))

        fused_columns = ["frame_no", "video_top_emotion", "video_top_confidence", "top_emotion", "top_confidence"] + COMMON_LABELS
        video_columns = ["frame_no", "top_emotion", "top_confidence"] + COMMON_LABELS
        audio_columns = ["window_no", "start_sec", "end_sec", "top_emotion", "top_confidence"] + COMMON_LABELS
        audio_chart_columns = ["window_no", "emotion", "probability"]
        fused_frame_df = fused_frame_df[fused_columns]
        video_frame_df = video_frame_df[video_columns]
        if audio_chart_df.empty:
            audio_chart_df = pd.DataFrame(columns=audio_chart_columns)
        else:
            audio_chart_df = audio_chart_df[audio_chart_columns]
        if not audio_window_df.empty:
            audio_window_df = audio_window_df[audio_columns]
        else:
            audio_window_df = pd.DataFrame(columns=audio_columns)
        chunk_columns = ["chunk_index", "chunk_text", "num_tokens", "weight", "top_emotion", "confidence", *COMMON_LABELS]
        if text_chunk_df.empty:
          text_chunk_df = pd.DataFrame(columns=chunk_columns)
        else:
          text_chunk_df = text_chunk_df[chunk_columns]
        return (
            fused_chart_df,
            fused_frame_df,
            stats_df,
            modality_df,
            video_chart_df,
            video_frame_df,
            audio_chart_df,
            audio_window_df,
            text_df,
          text_chunk_df,
            result["transcript"],
            _build_spotlight_html(result),
            result.get("preprocessed_video_path"),
            result,
        )

    def prepare_video_upload(video_path: str | None) -> tuple[str | None, str, str | None]:
        safe_path, status = analyzer.prepare_browser_safe_video(video_path)
        return safe_path, status, safe_path

    def stop_analysis() -> str:
        analyzer.request_cancel()
        return "Stop requested. The current model step will finish, then analysis will stop."

    with gr.Blocks(title="Video Emotion Frame Statistics") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
                <section class="hero-banner-fallback">
                  <div class="hero-banner-native">
                    <img src="{HEADER_BANNER_DATA_URI}" alt="EmoVision header banner" />
                  </div>
                </section>
                """,
                container=False,
                padding=False,
                elem_classes=["hero-banner-host"],
            )

            with gr.Row():
                with gr.Column(scale=4, elem_classes=["result-card"]):
                    video_input = gr.File(
                        label="Upload Video",
                        file_count="single",
                        file_types=[".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"],
                        type="filepath",
                    )
                    video_path_state = gr.State(value=None)
                    upload_preview = gr.Video(
                        label="Uploaded Video Preview",
                        format="mp4",
                        interactive=False,
                    )
                    video_status = gr.Textbox(
                        label="Video Status",
                        value="",
                        interactive=False,
                    )
                    transcript_override = gr.Textbox(
                        label="Transcript Override (Optional)",
                        lines=4,
                        placeholder="Leave blank to run ASR automatically.",
                    )
                    max_frames = gr.Slider(
                        minimum=24,
                        maximum=180,
                        step=12,
                        value=120,
                        label="Max Sampled Frames",
                    )
                    analyze_button = gr.Button("Analyze Emotions", elem_classes=["blue-button"], variant="primary")
                    stop_button = gr.Button("Stop Analysis", variant="stop")
                    transcript_quick_output = gr.Textbox(
                        label="Speech Transcript From Video",
                        lines=6,
                        interactive=False,
                    )
                with gr.Column(scale=5, elem_classes=["result-card"]):
                    info_output = gr.HTML(
                        "<section class='spotlight-card'><div class='spotlight-title'>Detected Overall Emotion</div><div class='spotlight-sub'>Upload a video to begin.</div></section>"
                    )
                    preprocessed_preview_side_output = gr.Video(
                        label="Pre-processed Video (Face Focus)",
                        format="mp4",
                        interactive=False,
                    )

            gr.HTML("<div id='statistics-section'></div>", container=False, padding=False)
            with gr.Tabs():
                with gr.Tab("All (Multimodal)"):
                    fused_plot_output = gr.LinePlot(
                        label="Multimodal Emotion Probability Over Frames",
                        x="frame_no",
                        y="probability",
                        color="emotion",
                        y_title="Probability",
                        x_title="Frames",
                    )
                    stats_output = gr.Dataframe(
                        headers=[
                            "emotion",
                            "mean_probability",
                            "max_probability",
                            "min_probability",
                            "std_probability",
                            "dominant_frames",
                            "dominant_ratio_percent",
                        ],
                        datatype=["str", "number", "number", "number", "number", "number", "number"],
                        interactive=False,
                        label="Aggregated Multimodal Statistics",
                    )
                    modality_output = gr.Dataframe(
                        headers=["modality", "status", "top_emotion", "confidence", "quality", "note"],
                        datatype=["str", "str", "str", "number", "number", "str"],
                        interactive=False,
                        label="Text, Audio & Video Modality Statistics",
                    )
                    fused_frame_output = gr.Dataframe(
                        interactive=False,
                        label="Per-Frame Multimodal Results",
                    )

                with gr.Tab("Video Only"):
                    video_plot_output = gr.LinePlot(
                        label="Video-Only Emotion Probability Over Frames",
                        x="frame_no",
                        y="probability",
                        color="emotion",
                        y_title="Probability",
                        x_title="Frames",
                    )
                    video_frame_output = gr.Dataframe(
                        interactive=False,
                        label="Per-Frame Video-Only Results",
                    )

                with gr.Tab("Audio Only"):
                    audio_plot_output = gr.LinePlot(
                        label="Audio-Only Emotion Probability Over Time Windows",
                        x="window_no",
                        y="probability",
                        color="emotion",
                        y_title="Probability",
                        x_title="Audio Windows",
                    )
                    audio_window_output = gr.Dataframe(
                        interactive=False,
                        label="Per-Window Audio-Only Results",
                    )

                with gr.Tab("Text Only"):
                    text_output = gr.Dataframe(
                        headers=["emotion", "probability"],
                        datatype=["str", "number"],
                        interactive=False,
                        label="Text-Only Emotion Distribution",
                    )
                    with gr.Accordion("Text Chunk Debug (Optional)", open=False):
                        text_chunk_output = gr.Dataframe(
                            headers=["chunk_index", "chunk_text", "num_tokens", "weight", "top_emotion", "confidence", *COMMON_LABELS],
                            datatype=["number", "str", "number", "number", "str", "number", *["number"] * len(COMMON_LABELS)],
                            interactive=False,
                            label="Text Chunk Details",
                        )

            with gr.Accordion("Debug Payload", open=False):
                raw_output = gr.JSON(label="Raw Analysis")

            video_input.upload(
                fn=prepare_video_upload,
                inputs=[video_input],
                outputs=[video_path_state, video_status, upload_preview],
            )

            analyze_event = analyze_button.click(
                fn=run_frame_statistics,
                inputs=[video_path_state, max_frames, transcript_override],
                outputs=[
                    fused_plot_output,
                    fused_frame_output,
                    stats_output,
                    modality_output,
                    video_plot_output,
                    video_frame_output,
                    audio_plot_output,
                    audio_window_output,
                    text_output,
                    text_chunk_output,
                    transcript_quick_output,
                    info_output,
                    preprocessed_preview_side_output,
                    raw_output,
                ],
            )
            stop_button.click(
                fn=stop_analysis,
                inputs=None,
                outputs=[video_status],
                cancels=[analyze_event],
                queue=False,
            )

    return demo
