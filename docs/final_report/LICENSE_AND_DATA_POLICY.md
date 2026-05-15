# License and Data Policy

This repository is prepared for public code review while keeping restricted data and artifacts local.

## What Can Be Public
- Source code and documentation.
- Configuration files (including experiment configs).
- Training and evaluation scripts.
- Dataset download/access instructions.
- Citation and license tables.

## What Must Stay Local
- Raw datasets.
- Preprocessed datasets or manifests derived from restricted datasets.
- Model weights, checkpoints, and runtime artifacts trained on NC or restricted datasets.
- Third-party local inference weights, including `artifacts/video_models/vit_based_fer_model`.
- Calibration outputs (unless explicitly approved for release).

## Non-Commercial Academic Use
This project is for academic, non-commercial use. Verify institutional policies and dataset licenses before use.

## Not Legal Advice
This document is informational only and is not legal advice. Always check the official license text.

## Final Defense Checklist
- Are all datasets and models cited?
- Are licenses and restrictions listed?
- Are NC/ND/restricted datasets excluded from public artifacts?
- Are model weights and artifacts local-only?
- Is the project described as academic/non-commercial?
- Is the video model described as a third-party integrated model?
- Is MELD described as calibration/evaluation only, not full fine-tune?
