# Data Access Guide

This guide lists public access points for datasets and license references. The repository does not download data, and all dataset files are local-only.

## General Rules
- Review each dataset license before access and use.
- Store datasets locally under data/raw or datasets (both are gitignored).
- Do not commit raw data, processed data, or derived manifests.

## Dataset Sources
- GoEmotions: https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/
- GoEmotions license: https://raw.githubusercontent.com/google-research/google-research/master/LICENSE
- RoBERTa-large model card: https://huggingface.co/FacebookAI/roberta-large
- Wav2Vec2/XLS-R model card: https://huggingface.co/facebook/wav2vec2-large-xlsr-53
- Wav2Vec2/SUPERB-ER model card: https://huggingface.co/superb/wav2vec2-base-superb-er
- CREMA-D: https://audeering.github.io/datasets/datasets/crema-d.html
- RAVDESS: https://zenodo.org/records/1188976
- TESS: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
- SAVEE registration: https://cvssp.org/savee/Register.html
- DailyDialog: https://huggingface.co/datasets/ConvLab/dailydialog
- EmotionLines: https://sites.google.com/view/emotionx2019/datasets
- MELD (project site): https://affective-meld.github.io/
- MELD (dataset card): https://huggingface.co/datasets/declare-lab/MELD
- ViT facial-expression model card: https://huggingface.co/mo-thecreator/vit-Facial-Expression-Recognition

## Local Handling
- Use the experiment configs under configs/experiments to record label mappings and planned outputs.
- Keep canonical label order: neutral, surprise, fear, sadness, joy, disgust, anger.
