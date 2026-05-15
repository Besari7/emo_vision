# Experiment Plan (License-Controlled)

All dataset files and trained artifacts are local-only and must not be committed to the repository.

Canonical label order:
["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

## Text Experiment
- Base artifact: artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7/best_model
- Previous fallback candidate: artifacts/text_models/roberta_large_goemotions_v2_clean_es/best_model
- Fine-tune target: DailyDialog balanced subset
- Output artifact: artifacts/text_models/roberta_large_goemotions_dailydialog_7class
- Label mapping:
  - no emotion -> neutral
  - happiness -> joy
  - sadness -> sadness
  - anger -> anger
  - fear -> fear
  - disgust -> disgust
  - surprise -> surprise
- Class imbalance handling: weighted CrossEntropy and/or WeightedRandomSampler
- EmotionLines: evaluation/analysis only due to CC BY-NC-ND / NoDerivatives risk; do not fine-tune.

## Audio Experiment
- Current inference artifact: artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop
- Previous artifact: artifacts/audio_models/wav2vec2_ravdess_7class
- Training sources: SAVEE, TESS, and RAVDESS aligned to the final seven-class label set
- Selection: validation macro-F1 early stopping
- TESS label mapping:
  - neutral -> neutral
  - pleasant surprise -> surprise
  - fear -> fear
  - sadness -> sadness
  - happiness -> joy
  - disgust -> disgust
  - anger -> anger
- The current model's native label order is angry, disgust, fear, happy, neutral, sad, surprise; convert outputs to the canonical order before fusion.
- SAVEE use requires registration/license approval; dataset files and derived artifacts remain local-only.

## Video
- Use mo-thecreator/vit-Facial-Expression-Recognition as a third-party integrated inference model.
- Do not fine-tune and do not commit weights.
- Document license uncertainty clearly.

## MELD Calibration (Evaluation Only)
- Do not use MELD for full fine-tuning.
- Allowed: temperature scaling, class-specific bias, optional fusion weight search.
- Calibration outputs are local artifacts unless explicitly approved.
