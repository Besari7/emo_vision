# Project Final Scope

This document defines the final capstone scope for report writing, evidence collection, and cleanup review.

## Text Branch

The final text branch is RoBERTa-large fine-tuned on GoEmotions. The current local inference artifact is `roberta_large_goemotions_ekman_v2_continued_from_direct7`.

MELD adaptation was attempted during development, but it was stopped early because class imbalance and poor validation behavior made it unsuitable for final reporting.

BERT references in the repository are legacy/prototype or checkpoint-compatibility references. They must not be presented as final model claims unless a specific file is explicitly documenting historical development work.

## Audio Branch

The final audio branch is Wav2Vec2/XLS-R based emotion recognition.

The final audio inference artifact is `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`, a seven-class Wav2Vec2/XLS-R classifier aligned to the canonical label set.

The previous audio artifact, `wav2vec2_ravdess_7class`, is no longer the default inference path. It should only be described as historical or fallback material if explicitly needed.

The final canonical emotion labels are:

1. `neutral`
2. `surprise`
3. `fear`
4. `sadness`
5. `joy`
6. `disgust`
7. `anger`

## Visual Branch

The final visual branch is a ViT-based facial-expression recognition branch.

The local inference artifact path is `artifacts/video_models/vit_based_fer_model`. It is treated as a third-party integrated inference model; weights remain local-only and are not distributed in the GitHub repository.

Additional ViT fine-tuning was attempted during development, but MELD-related class imbalance caused poor early behavior. The final visual work focuses on integration, inference pipeline adaptation, label/probability alignment, backend/frontend compatibility, demo support, and fusion compatibility.

Do not claim that the visual branch is a fully re-trained project-specific visual model.

## Fusion

Fusion operates on modality probability outputs. Text, audio, and visual probabilities must be aligned to the canonical label order before fusion:

`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`

Final fusion metrics must only be reported if generated from actual final repository outputs. Do not invent metrics, reuse placeholder outputs, or treat external model-card scores as project-owned final results.

MELD results are domain-shift diagnostics, not final fusion calibration. Final fusion calibration should be done on a target-domain validation manifest, and MELD grid-search outputs must not be applied automatically to the demo or to `configs/final_capstone.json`.
