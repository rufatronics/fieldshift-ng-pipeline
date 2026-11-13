# FieldShift NG pipeline

This repository runs the FieldShift NG NASA data pipeline and publishes the resulting private dataset to Hugging Face. It supports the original three-zone run and a Nigeria-wide state/FCT run using 37 representative state coordinates.

## Required GitHub Actions secrets

- `HF_TOKEN`: Hugging Face token with write access.
- `EARTHDATA_TOKEN`: NASA Earthdata token.

Add them under **Settings → Secrets and variables → Actions**. Never commit tokens to this repository.

## Running

The workflow supports manual dispatch with `test_mode=true` by default. Choose `three_zones` to use Kano, Katsina, and Sokoto, or `nigeria_states` to use all 36 states plus the Federal Capital Territory. The national run uploads to `rufatronics/fieldshift-ng-data-nigeria`; the three-zone production run uploads to `rufatronics/fieldshift-ng-data`. The scheduled run uses the full three-zone configuration. The pipeline caches downloaded data between runs and uploads only after a successful local harmonization.
