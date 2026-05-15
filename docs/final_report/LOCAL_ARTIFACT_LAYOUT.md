# Local Artifact Layout

Model artifacts are local files and should remain under `artifacts/`.

## Final Inference Artifacts

- Audio: `artifacts/audio_models/wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`
- Text: `artifacts/text_models/roberta_large_goemotions_ekman_v2_continued_from_direct7`
- Video: `artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition`

Audio and text keep loadable model files under `best_model/`; the demo resolves those folders automatically.

The final audio inference artifact is `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`, a seven-class Wav2Vec2/XLS-R classifier aligned to the canonical label set.

The previous text artifact, `artifacts/text_models/roberta_large_goemotions_v2_clean_es`, remains a local fallback candidate if the text branch is rolled back. The previous audio artifact, `artifacts/audio_models/wav2vec2_ravdess_7class`, is no longer the default inference path.

## Canonical Labels

`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`
