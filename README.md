# EmoVision

EmoVision is a local multimodal emotion recognition project for video-based analysis. It combines text, audio, and visual emotion signals into a canonical seven-class emotion space and exposes the final workflow through a Gradio demo.

The repository is structured as an installable Python package with supporting scripts for local demo execution, model smoke tests, inference evaluation, fusion grid search, and final-report evidence checks.

Repository: `https://github.com/Besari7/emo_vision`

## Final Project Scope

The final capstone scope is:

- **Text:** RoBERTa-large fine-tuned on GoEmotions and aligned to the final seven-class label set.
- **Audio:** Wav2Vec2/XLS-R emotion classifier trained for the canonical seven labels.
- **Video:** ViT-based facial-expression inference integration.
- **Fusion:** probability-level multimodal fusion over canonical label-aligned outputs.

Legacy or prototype references to older BERT, EfficientNet, old label orders, or previous artifact names are retained only for compatibility review and development traceability. They should not be interpreted as final model claims unless a file explicitly describes them as such.

## Emotion Labels

All active inference and fusion code uses the following canonical label order:

```text
neutral, surprise, fear, sadness, joy, disgust, anger
```

Label normalization and probability reordering are implemented in `src/multimodal_emotion/labels.py` to prevent modality outputs from being fused with mismatched class indices.

## Demo Features

The local demo accepts a video upload and produces:

- overall multimodal emotion prediction
- modality-level text, audio, and video summaries
- frame-level multimodal emotion statistics
- video-only frame probability timelines
- audio-only window probability timelines
- text-only probability distribution and optional chunk diagnostics
- transcript output from Whisper ASR or a manual transcript override
- face-focused preprocessed video preview for visual inference inspection

The primary entrypoint is:

```powershell
python scripts/run_demo.py
```

Default local URL:

```text
http://127.0.0.1:7860
```

Useful runtime options:

```powershell
python scripts/run_demo.py --host 127.0.0.1 --port 7860
python scripts/run_demo.py --no-preload-models
python scripts/run_demo.py --share
```

## Quick Start

Python 3.10 or newer is required.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-demo.txt
python scripts/final_report/check_local_artifacts.py
python scripts/final_report/test_model_loading.py --all
python scripts/run_demo.py
```

For the full development dependency set:

```powershell
pip install -r requirements.txt
python -m pytest
```

## Required Local Artifacts

Model weights and dataset-derived artifacts are intentionally not committed to this repository. Place the final local artifacts under:

```text
artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7/
artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop/
artifacts/video_models/vit_based_fer_model/
assets/ui/header_emovision.png
```

The text and audio packages may keep loadable Hugging Face model files inside a nested `best_model/` directory. The runtime resolver automatically checks that fallback location when the package root does not directly contain model files.

The ASR branch uses `openai/whisper-small.en`. If the model is not already available in the local Hugging Face cache, use the transcript override field in the UI or cache the ASR model before running automatic transcription.

## Runtime Configuration

The final project configuration is stored in:

```text
configs/final_capstone.json
```

Default inference paths and runtime settings are resolved by `src/multimodal_emotion/inference/runtime_config.py`.

Model paths can be overridden with environment variables:

```powershell
$env:EMOVISION_TEXT_MODEL="C:\path\to\text-model-or-best_model"
$env:EMOVISION_AUDIO_MODEL="C:\path\to\audio-model-or-best_model"
$env:EMOVISION_VIDEO_MODEL="C:\path\to\video-model"
```

Temperature overrides are also supported:

```powershell
$env:EMOVISION_TEXT_TEMPERATURE="1.0"
$env:EMOVISION_AUDIO_TEMPERATURE="1.4"
$env:EMOVISION_VIDEO_TEMPERATURE="1.3"
```

Current default fusion weights are:

```text
text: 0.55
audio: 0.20
video: 0.25
```

## Repository Layout

```text
src/multimodal_emotion/     Python package for data, models, inference, evaluation, export, and demo code
scripts/                    Local utilities for demo, evaluation, diagnostics, and final-report checks
configs/                    Final project config and experiment definitions
docs/final_report/          Scope, setup, data policy, artifact, and reporting documentation
assets/ui/                  UI assets used by the demo
reports/final_report_evidence/
                            Evidence folders generated for the final report
artifacts/                  Local model files and runtime outputs; not committed
archive/                    Historical/prototype material retained for review
tests/                      Unit and contract tests
```

## Evaluation and Diagnostics

Smoke-test local artifact availability without running the full demo:

```powershell
python scripts/final_report/check_local_artifacts.py
python scripts/final_report/test_model_loading.py --text --audio --video
```

Evaluate a manifest with one modality or fused inference:

```powershell
python scripts/evaluate_inference.py `
  --manifest path\to\manifest.jsonl `
  --modality fusion `
  --output-dir reports\eval_fusion
```

Run fusion grid search on validation data only:

```powershell
python scripts/grid_search_fusion.py `
  --manifest path\to\validation_manifest.jsonl `
  --output-dir reports\fusion_grid_search `
  --fusion-mode both
```

Do not tune fusion on a held-out test set. Grid-search outputs are validation or domain-specific recommendations and are not applied automatically to the final runtime configuration.

## Data and License Policy

Datasets, trained weights, checkpoints, and large generated runtime outputs remain local-only. This repository is intended for academic, non-commercial capstone work and does not redistribute dataset-derived model artifacts.

Policy and reporting references:

- `docs/final_report/PROJECT_FINAL_SCOPE.md`
- `docs/final_report/LOCAL_SETUP_WINDOWS.md`
- `docs/final_report/LOCAL_ARTIFACT_LAYOUT.md`
- `docs/final_report/LICENSE_AND_DATA_POLICY.md`
- `docs/final_report/DATA_ACCESS_GUIDE.md`
- `docs/final_report/THIRD_PARTY_NOTICES.md`
- `docs/final_report/FINAL_EVIDENCE_CHECKLIST.md`

## Notes on Report Claims

Only metrics generated from the final repository outputs should be reported as project-owned results. External model-card metrics, placeholder values, or legacy prototype outputs must not be reused as final project metrics.
