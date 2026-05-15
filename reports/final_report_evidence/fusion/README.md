# Evidence Folder

Place verified final capstone evidence in this folder only after it has been generated from actual final outputs.

Do not store invented metrics, placeholder predictions, or external model-card scores as project-owned results.

MELD results are domain-shift diagnostics, not final fusion calibration. Final fusion calibration should be done on a target-domain validation manifest.

Keep `artifacts/fusion_search/meld_small` as a report/analysis artifact only. Do not copy its `best_fusion_config.json` into `configs/final_capstone.json`, and do not use it to change demo default weights.
