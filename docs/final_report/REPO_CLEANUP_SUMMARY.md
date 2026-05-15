# Repository Cleanup Summary

Date inspected: 2026-05-11
Model default update: 2026-05-15

Scope used for this review:

- Text branch final scope: RoBERTa-large fine-tuned on GoEmotions. Current local artifact: `roberta_large_goemotions_ekman_v2_continued_from_direct7`.
- Audio branch final scope: Wav2Vec2/XLS-R based emotion recognition. Current local artifact: `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop`.
- Visual branch final scope: ViT-based facial-expression recognition integration, inference adaptation, label/probability alignment, backend/frontend compatibility, demo support, and fusion compatibility.
- Canonical label order: `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`.
- Fusion requirement: all modality probability outputs must be aligned to the canonical label order before fusion.
- Reporting rule: do not present invented metrics, fake confusion matrices, external model-card scores, or prototype results as project-owned final results.

## Files That Appear Final/Current

- `README.md`
  - Describes the active local EmoVision demo.
  - Points to the current demo entrypoint, service, UI, fusion code, and expected local artifact paths.
  - Correctly lists the ViT visual artifact path: `artifacts/video_models/mo-thecreator-vit-Facial-Expression-Recognition/`.
  - Needs final wording expansion, but appears aligned with the active demo workflow.

- `scripts/run_demo.py`
  - Active launcher for the Gradio demo.
  - Imports `MultimodalDemoAnalyzer` and `build_demo`.
  - Should be retained.

- `src/multimodal_emotion/demo/ui.py`
  - Active Gradio frontend.
  - Uses `COMMON_LABELS` from the demo service/fusion layer.
  - Appears current for demo support and frontend compatibility.

- `src/multimodal_emotion/demo/service.py`
  - Main active backend service for ASR, text/audio/video HF pipelines, face-focused video processing, audio/video timelines, preview generation, and multimodal fusion support.
  - Contains current local artifact paths for text, audio, and video artifacts.
  - Includes ViT image-classification path for the visual branch.
  - Should be retained as the active demo backend.

- `src/multimodal_emotion/demo/fusion.py`
  - Active probability remapping and weighted fusion utility for the demo.
  - Contains modality label maps and probability normalization.
  - Uses the final canonical label order from `src/multimodal_emotion/labels.py`.

- `src/multimodal_emotion/config.py`
  - Default project configuration is aligned with the final RoBERTa/canonical-label architecture.
  - Explicit legacy config files can still override these defaults when loading older checkpoints.

- `src/multimodal_emotion/inference/ensemble.py`
  - Loads weighted ensemble members from local artifacts and produces probability outputs.
  - Includes validation that ensemble members share the same labels.
  - Should be retained until artifact compatibility is verified.

- `src/multimodal_emotion/evaluation/metrics.py`
  - Provides metric computation utilities from real predictions.
  - Does not appear to contain hard-coded/fake results.
  - Keep as utility code, but final reporting should only include metrics generated from verified final experiments.

- `assets/ui/header_emovision.png`
  - Required by README as the demo header asset.
  - Should be retained.

## Files That Appear Legacy/Prototype

- `src/multimodal_emotion/models/fusion.py`
  - Multimodal model code still imports and supports BERT random initialization for explicit legacy configs:
    - `BertConfig`, `BertModel`
    - fallback branch for `"bert"` text model names
  - May be useful for loading old checkpoints, but should not be presented as the final text branch.

- `src/multimodal_emotion/data/synthetic.py`
  - Synthetic toy data generator.
  - Useful for development or smoke tests, but it should not be used as final evidence or final performance reporting.

- `src/multimodal_emotion/export/onnx_export.py`
  - Uses a placeholder sentence for dummy export input.
  - This is acceptable as export scaffolding, but should not appear in final report evidence as a real inference example.

## Files That Need Manual Review

- `src/multimodal_emotion/demo/fusion.py`
  - Keep modality maps synchronized with labels emitted by the active text, audio, and video artifacts.
  - Any new model label must be explicitly mapped into the canonical seven-label set before fusion.

- `src/multimodal_emotion/config.py`
  - Defaults now match the final model description and canonical label order.
  - Legacy checkpoints should provide explicit config files instead of relying on defaults.

- `src/multimodal_emotion/inference/ensemble.py`
  - It trusts each checkpoint/config label order and rejects mismatches between ensemble members.
  - Manual review is needed to ensure ensemble probabilities are reordered into the final canonical label order before fusion/reporting.

- `src/multimodal_emotion/demo/service.py`
  - Current active service uses final-style HF text/audio/video classifiers and runtime media preprocessing.
  - Review only when adding new model artifacts or changing modality weights.

- `src/multimodal_emotion/models/fusion.py`
  - Keep the BERT/Roberta branches for checkpoint compatibility.
  - Final documentation should describe RoBERTa, Wav2Vec2/XLS-R, ViT, and aligned probability fusion.

- `src/multimodal_emotion/training/engine.py`
  - Computes validation metrics and confusion records dynamically.
  - Review before using any generated output in the final report; only verified final runs should be cited.

- `src/multimodal_emotion/evaluation/metrics.py`
  - Metric utilities are legitimate, but final report content must avoid placeholder or unverified results.

- `requirements.txt` and `pyproject.toml`
  - Dependencies still include `torchvision`, which may be required by installed model/image stacks.
  - Do not remove until the active demo path is tested without it.

- Empty evidence/report directories:
  - `archive/legacy/`
  - `configs/`
  - `docs/final_report/`
  - `reports/final_report_evidence/audio/`
  - `reports/final_report_evidence/demo/`
  - `reports/final_report_evidence/fusion/`
  - `reports/final_report_evidence/repo/`
  - `reports/final_report_evidence/report_notes/`
  - `reports/final_report_evidence/text/`
  - `reports/final_report_evidence/video/`
  - `scripts/final_report/`
  - These may be intended staging locations. Do not delete yet.

## Suggested Changes

1. Standardize canonical labels everywhere.
   - Final order should be: `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`.
   - Current final-facing code uses this order via `src/multimodal_emotion/labels.py`.
   - Keep explicit probability reordering at every modality boundary before fusion.

2. Update text-branch descriptions.
   - Replace final-facing BERT wording with RoBERTa-large fine-tuned on GoEmotions.
   - Keep BERT references only if clearly marked as legacy/prototype/checkpoint-compatibility code.
   - Update `pyproject.toml` description.

3. Review text defaults.
   - Defaults now point to RoBERTa-large.
   - If legacy checkpoints require BERT defaults, isolate that behavior behind explicit legacy config files.

4. Clean final visual-branch wording.
   - Final-facing documentation should describe ViT-based facial-expression recognition integration and inference adaptation.
   - Keep any future non-ViT visual code clearly marked as legacy/prototype compatibility.

5. Update audio-branch wording.
   - Final-facing documentation should describe `wav2vec2_xlsr_savee_tess_ravdess_rf_style_earlystop` as the final audio inference artifact.
   - Treat `wav2vec2_ravdess_7class` as the previous audio artifact, not the default inference path.
   - Distinguish generic Wav2Vec2/XLS-R feature extraction from the final trained audio classifier artifact.

6. Audit final report evidence.
   - Do not include synthetic dataset outputs, placeholder ONNX export examples, unverified validation logs, or external model-card scores as project-owned final results.
   - Only include metrics from verified final project runs with known datasets, splits, labels, and artifact versions.

7. Add a small label-alignment test before cleanup.
   - Test that text, audio, video, ensemble, and fusion outputs all expose probabilities in the canonical order.
   - Include alias coverage for GoEmotions-style text labels, audio dataset labels, and facial-expression labels.

## Risky Files That Should Not Be Deleted Yet

- `src/multimodal_emotion/demo/service.py`
  - Contains active demo logic and runtime media preprocessing.
  - Deleting or aggressively editing it could break the demo.

- `src/multimodal_emotion/demo/fusion.py`
  - Central to probability remapping and fusion.
  - Needs careful extension when adding new external model labels.

- `src/multimodal_emotion/config.py`
  - Defaults are final-aligned, but checkpoint loading and training utilities still depend on this schema.

- `src/multimodal_emotion/models/fusion.py`
  - Contains legacy BERT compatibility, but may be required by saved ensemble members.

- `src/multimodal_emotion/inference/ensemble.py`
  - Required for loading ensemble artifacts listed by the README.

- `src/multimodal_emotion/evaluation/metrics.py`
  - Useful for verified evaluation; risk is misuse in reporting, not the utility itself.

- `src/multimodal_emotion/data/synthetic.py`
  - Development-only, but may be used by smoke tests or examples.

- `requirements.txt` / `pyproject.toml`
  - Dependency cleanup should wait until after active demo and artifact loading are tested.

- `archive/legacy/`
  - Empty at inspection time, but intentionally named as legacy storage.
  - Do not delete until the team confirms it is unused.

- `reports/final_report_evidence/`
  - Empty category directories appear intentionally prepared for final evidence.
  - Do not delete.

## Search Findings

Key compatibility/prototype references found:

- BERT:
  - `src/multimodal_emotion/models/fusion.py`

- Placeholder text:
  - `src/multimodal_emotion/export/onnx_export.py`

- Confusion matrix / metrics code:
  - `src/multimodal_emotion/evaluation/metrics.py`
  - `src/multimodal_emotion/training/engine.py`
  - `src/multimodal_emotion/inference/ensemble.py`

No hard-coded fake metric values or fake confusion matrices were found in the inspected committed source files. Metric utilities generate values dynamically from predictions and labels, but their outputs still require manual verification before final report use.

## Stage 2 Changes Applied

Date applied: 2026-05-11

### Files Created

- `docs/final_report/PROJECT_FINAL_SCOPE.md`
- `docs/final_report/LABEL_ALIGNMENT.md`
- `docs/final_report/FINAL_EVIDENCE_CHECKLIST.md`
- `configs/final_capstone.json`
- `reports/final_report_evidence/repo/final_label_mapping.json`
- `src/multimodal_emotion/labels.py`
- `scripts/final_report/build_final_evidence.py`
- `scripts/final_report/compute_metrics_from_predictions.py`
- `scripts/final_report/check_label_alignment.py`
- `reports/final_report_evidence/audio/README.md`
- `reports/final_report_evidence/demo/README.md`
- `reports/final_report_evidence/fusion/README.md`
- `reports/final_report_evidence/text/README.md`
- `reports/final_report_evidence/video/README.md`
- `reports/final_report_evidence/repo/git_commit.txt`
- `reports/final_report_evidence/repo/repo_file_inventory.tsv`

### Files Edited

- `README.md`
  - Added a final capstone scope note warning that legacy/prototype components remain in the repository.

- `pyproject.toml`
  - Replaced the outdated BERT-based description with a neutral final-capstone description.

- `src/multimodal_emotion/demo/fusion.py`
  - Switched `COMMON_LABELS` to the canonical label order.
  - Imported central label helpers from `src/multimodal_emotion/labels.py`.
  - Added explicit canonical probability reordering before weighted fusion.
  - Preserved public function names used by the demo.

- `src/multimodal_emotion/demo/service.py`
  - Kept the ViT HF image-classification path intact as the active visual inference path.

- `docs/final_report/REPO_CLEANUP_SUMMARY.md`
  - Appended this Stage 2 change log.

### Verification Performed

- Python syntax compilation completed for `src` and `scripts/final_report`.
- `scripts/final_report/build_final_evidence.py` ran and prepared evidence folders.
- `scripts/final_report/check_label_alignment.py` originally reported a legacy mismatch in `src/multimodal_emotion/config.py`; this was later corrected in Stage 3.
- `scripts/final_report/compute_metrics_from_predictions.py --help` runs successfully. Full metric generation still requires real `.npy` prediction/label inputs and installed project dependencies.

### Files Intentionally Not Changed

- `src/multimodal_emotion/models/fusion.py`
  - Still contains BERT/Roberta compatibility code. This may be required for older checkpoints and was not removed.

- `src/multimodal_emotion/inference/ensemble.py`
  - Not changed because artifact label-order compatibility needs manual verification before migration.

- `requirements.txt`
  - Not changed because `torchvision` may still be required by compatibility paths and image preprocessing.

- Source files were not deleted.
- Legacy compatibility code was not removed.

### Remaining Manual Review Items

- Confirm that text, audio, video, and ensemble outputs are converted to canonical order before final report fusion.
- Generate final evidence files from actual final outputs before reporting metrics.
- Run the final demo after local model artifacts are available to confirm the stricter label mapping covers all emitted labels.

## Stage 3 Changes Applied

Date applied: 2026-05-12

- `src/multimodal_emotion/config.py`
  - Changed default labels to use `CANONICAL_LABELS`.
  - Changed default text model metadata from BERT to RoBERTa-large.
  - Kept schema compatibility so explicit legacy configs can still load older checkpoint settings.

- Verification:
  - `scripts/final_report/check_label_alignment.py` now reports all checked label lists as `OK`.
  - Python syntax compilation completed for `src` and `scripts`.

## Stage 4 Changes Applied

Date applied: 2026-05-12

- `scripts/final_report/check_label_alignment.py`
  - Added local artifact `id2label` checks for Hugging Face `config.json` files under `artifacts/`.
  - Artifact labels are normalized through the canonical alias map, so values such as `happy` and `sad` are checked as `joy` and `sadness`.
  - Non-canonical artifact output order is reported as `ARTIFACT_CONVERT` instead of being hidden by the source-config checks.

- `scripts/final_report/compute_metrics_from_predictions.py`
  - Added `--model-config` support to read artifact-native `id2label` order.
  - Integer class ids and 2D probability arrays now require `--model-config` or explicit `--label-names`, preventing silent metric shifts.
  - Metric output records the label source used.

- `configs/final_capstone.json`
  - Replaced the provisional fusion type with `text_reduced_equal_weight_probability_fusion`.
  - Recorded the active demo implementation and the intentional text weight reduction ratio.

- Runtime cleanup:
  - Removed removable items under `artifacts/gradio_tmp`, `artifacts/runtime_previews`, and `artifacts/runtime_workspaces`.
  - Some locked/protected runtime directories may require closing active processes before deletion.
