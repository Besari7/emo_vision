# Local Setup on Windows

This guide uses PowerShell from the repository root:

```powershell
cd C:\Users\berke\multimodel_emotion_recognition-main
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

- Audio: `artifacts/audio_models/wav2vec2_ravdess_7class`
- Text: `artifacts/text_models/roberta_large_goemotions_v2_clean_es`
- Video: `artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition`

Audio and text keep loadable model files under `best_model/`; the demo and smoke-test scripts resolve those folders automatically.

The final audio model, `wav2vec2_ravdess_7class`, was obtained by further fine-tuning the CREMA-D fine-tuned Wav2Vec2/SUPERB-ER checkpoint on RAVDESS using the final seven-class label set.

CREMA-D checkpoints are the first fine-tuning stage / initialization checkpoint for the final RAVDESS model.

Large model files must stay under `artifacts/` and must not be pushed to GitHub.

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
