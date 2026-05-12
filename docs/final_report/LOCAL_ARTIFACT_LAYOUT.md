# Local Artifact Layout

Model artifacts are local files and should remain under `artifacts/`.

## Final Inference Artifacts

- Audio: `artifacts/audio_models/wav2vec2_ravdess_7class`
- Text: `artifacts/text_models/roberta_large_goemotions_v2_clean_es`
- Video: `artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition`

Audio and text keep loadable model files under `best_model/`; the demo resolves those folders automatically.

The final audio model, `wav2vec2_ravdess_7class`, was obtained by further fine-tuning the CREMA-D fine-tuned Wav2Vec2/SUPERB-ER checkpoint on RAVDESS using the final seven-class label set.

CREMA-D checkpoints, including paths such as `artifacts/audio_models/wav2vec2_cremad_bs16_lr3e5`, are the first fine-tuning stage / initialization checkpoint for the final RAVDESS seven-class model. They are not the final inference path.

## Canonical Labels

`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`
