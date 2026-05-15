# Label Alignment

## Canonical Label Order

The final capstone label order is:

1. `neutral`
2. `surprise`
3. `fear`
4. `sadness`
5. `joy`
6. `disgust`
7. `anger`

All final probability vectors should use this order unless a file explicitly states that it is a legacy artifact requiring conversion.

## Why Alignment Matters

Fusion combines probability outputs from text, audio, and visual branches. If two branches use the same label names but different vector indices, fusion will mix probabilities for the wrong emotions.

For example, a legacy vector order of `neutral`, `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise` places `joy` at index 1. In the final canonical order, index 1 is `surprise`. If this vector is fused without conversion, `joy` probability mass can be interpreted as `surprise`.

## Required Conversion Before Fusion

Each modality should expose probabilities as a mapping from label name to score, then reorder that mapping into the canonical order before vector-level fusion.

Text outputs should be normalized from GoEmotions-style labels into canonical labels. For example, `happy` and `happiness` map to `joy`.

Audio outputs should normalize common dataset abbreviations and labels. For example, `neu` maps to `neutral`, `hap` maps to `joy`, `ang` maps to `anger`, and `angry` maps to `anger`.

The final audio inference artifact is `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`, a seven-class Wav2Vec2/XLS-R classifier aligned to the canonical label set. Its native label order is `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`, so outputs must be normalized before fusion.

Visual outputs should normalize facial-expression labels and aliases. For example, `angry` maps to `anger`, `fearful` maps to `fear`, and `surprised` maps to `surprise`.

## Legacy Config Risk

Legacy configs may contain older label orders and must not be used as the final fusion order without conversion.

Known older order seen during cleanup:

`neutral`, `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`

When loading older configs or checkpoints, preserve compatibility during loading, then explicitly convert output probabilities into the final canonical order before final reporting or fusion.

## Metric Generation

Saved `.npy` prediction and label arrays may contain integer class ids in the artifact's native output order. Do not assume those integer ids are already canonical.

For integer class ids or 2D probability arrays, run `scripts/final_report/compute_metrics_from_predictions.py` with either `--model-config path/to/config.json` or an explicit `--label-names` list in model output index order. The script intentionally rejects integer/probability inputs without an explicit label source.
