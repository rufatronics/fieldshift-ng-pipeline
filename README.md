# FieldShift NG pipeline

This repository runs the FieldShift NG NASA data pipeline and publishes the resulting private dataset to Hugging Face.

## Required GitHub Actions secrets

- `HF_TOKEN`: Hugging Face token with write access.
- `EARTHDATA_TOKEN`: NASA Earthdata token.

Add them under **Settings → Secrets and variables → Actions**. Never commit tokens to this repository.

## Running

The workflow supports manual dispatch with `test_mode=true` by default. The scheduled run uses the full configured period. The pipeline caches downloaded data between runs and uploads only after a successful local harmonization.
