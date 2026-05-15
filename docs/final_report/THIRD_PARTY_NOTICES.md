# Third-Party Notices

This table is intended to be public and to support citations and license tracking. Always verify the latest license terms at the source.

| Component | Role | Source | License / Restriction | Use in Project | Public Distribution Policy | Citation Required |
| --- | --- | --- | --- | --- | --- | --- |
| GoEmotions | Text dataset | https://research.google/pubs/goemotions-a-dataset-of-fine-grained-emotions/<br>https://raw.githubusercontent.com/google-research/google-research/master/LICENSE | Google Research license; review terms | Base text fine-tune (existing model lineage) | Dataset files not distributed; derived weights local-only | Yes |
| RoBERTa-large | Text backbone | https://huggingface.co/FacebookAI/roberta-large | See model card; license terms apply | Text encoder for GoEmotions model | Weights not redistributed; local-only | Yes |
| Wav2Vec2/XLS-R | Audio backbone | https://huggingface.co/facebook/wav2vec2-large-xlsr-53 | See model card; license terms apply | Audio backbone for current local audio artifact | Weights not redistributed; local-only | Yes |
| Wav2Vec2/SUPERB-ER | Audio backbone | https://huggingface.co/superb/wav2vec2-base-superb-er | See model card; license terms apply | Previous audio lineage | Weights not redistributed; local-only | Yes |
| CREMA-D | Audio dataset | https://audeering.github.io/datasets/datasets/crema-d.html | Research/academic use; review terms | Previous audio lineage | Dataset files not distributed; derived weights local-only | Yes |
| RAVDESS | Audio dataset | https://zenodo.org/records/1188976 | Review Zenodo license; non-commercial restrictions may apply | Current audio training/evaluation source | Dataset files not distributed; derived weights local-only | Yes |
| TESS | Audio dataset | https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF | Review dataset terms | Current audio training/evaluation source | Dataset files not distributed; derived weights local-only | Yes |
| SAVEE | Audio dataset | https://cvssp.org/savee/Register.html | Registration required; license terms after approval | Current audio training/evaluation source | Do not distribute dataset or derived weights | Yes |
| DailyDialog | Text dataset | https://huggingface.co/datasets/ConvLab/dailydialog | Review dataset card; non-commercial restrictions may apply | Planned fine-tune target | Dataset files not distributed; derived weights local-only | Yes |
| EmotionLines | Text dataset | https://sites.google.com/view/emotionx2019/datasets | CC BY-NC-ND / NoDerivatives risk; evaluation only | Optional evaluation/analysis only | Dataset files not distributed | Yes |
| MELD | Multimodal dataset | https://affective-meld.github.io/<br>https://huggingface.co/datasets/declare-lab/MELD | Review dataset terms | Calibration/evaluation only, not full fine-tune | Dataset files not distributed; calibration outputs local-only | Yes |
| vit_based_fer_model | Video model | Local-only (source link removed) | License uncertain; verify model card if reintroduced | Third-party integrated inference model | Weights not redistributed; local-only | Yes |
