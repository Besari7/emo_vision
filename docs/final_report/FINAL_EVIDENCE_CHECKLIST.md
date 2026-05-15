# Final Evidence Checklist

Use this checklist to collect final report evidence. Files should be generated from actual final outputs only. Do not fabricate metrics, confusion matrices, or predictions.

## Text Evidence

When generating metrics from integer class ids or probability arrays, pass the text artifact config as the label source:

Current text artifact: `artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7`

If integer class ids or probability arrays are exported for this artifact, use:

`python scripts/final_report/compute_metrics_from_predictions.py --preds PATH_TO_TEXT_PREDS.npy --labels PATH_TO_TEXT_LABELS.npy --model-config artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7/best_model/config.json --output-dir reports/final_report_evidence/text`

- `text/test_metrics.json`
- `text/test_preds.npy`
- `text/test_labels.npy`
- `text/classification_report.csv`
- `text/confusion_matrix.png`

## Audio Evidence

The final audio inference artifact is `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`, a seven-class Wav2Vec2/XLS-R classifier aligned to the canonical label set.

When generating metrics from integer class ids or probability arrays, pass the audio artifact config as the label source.

- `audio/test_metrics.json`
- `audio/test_preds.npy`
- `audio/test_labels.npy`
- `audio/classification_report.csv`
- `audio/confusion_matrix.png`

## Video Evidence

- `video/video_branch_summary.md`
- `video/demo_predictions.csv`
- `video/screenshots/`

## Fusion Evidence

- `fusion/fusion_method.md`
- `fusion/test_predictions.csv`
- `fusion/test_metrics.json`
- `fusion/classification_report.csv`
- `fusion/confusion_matrix.png`

## Demo Evidence

- `demo/screenshots/`
