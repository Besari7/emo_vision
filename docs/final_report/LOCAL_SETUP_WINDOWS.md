# Local Setup on Windows

This guide uses PowerShell from the repository root:

```powershell
git clone https://github.com/Besari7/emo_vision.git
cd emo_vision
```

## 1. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, allow scripts for the current terminal process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install Demo Dependencies

```powershell
pip install -r requirements-demo.txt
```

If you see `ModuleNotFoundError: No module named 'transformers'`, the active environment does not have the demo dependencies installed. Activate `.venv` and run:

```powershell
pip install -r requirements-demo.txt
```

CUDA is not required for these smoke tests. CPU inference is acceptable but may be slow.

## 3. Check Local Model Artifacts

```powershell
python scripts/final_report/check_local_artifacts.py
```

Expected local artifact paths:

- Audio: `artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`
- Text: `artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7`
- Video: `artifacts/video_models/vit_based_fer_model`

Audio and text keep loadable model files under `best_model/`; the demo and smoke-test scripts resolve those folders automatically.

The final audio inference artifact is `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`, a seven-class Wav2Vec2/XLS-R classifier aligned to the canonical label set.

The previous text artifact, `artifacts/text_models/roberta_large_goemotions_v2_clean_es`, remains a local fallback candidate if the text branch is rolled back. The previous audio artifact, `artifacts/audio_models/wav2vec2_ravdess_7class`, is no longer the default inference path.

Large model files must stay under `artifacts/` and must not be pushed to GitHub. The local video model directory is `artifacts/video_models/vit_based_fer_model`; its weights are required for local inference but are not part of the public repository.

## 4. Smoke Test Model Loading

Text:

```powershell
python scripts/final_report/test_model_loading.py --text
```

Video:

```powershell
python scripts/final_report/test_model_loading.py --video
```

Audio, only if a WAV file is available:

```powershell
python scripts/final_report/test_model_loading.py --audio --audio-file "C:\path\to\sample.wav"
```

The loading script is a smoke test, not final evaluation.

## 5. Run the Demo

```powershell
python scripts/run_demo.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'transformers'`: activate `.venv`, then run `pip install -r requirements-demo.txt`.
- PowerShell `ExecutionPolicy` blocks activation: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.venv` again.
- CPU is fine for smoke tests; CUDA is optional and may only improve speed.
- Keep large model files under `artifacts/`. Do not commit `.safetensors`, `.bin`, `.pt`, or `.pth` files.
