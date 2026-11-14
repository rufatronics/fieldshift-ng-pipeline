# FieldShift NG pipeline

This repository runs the FieldShift NG NASA data pipeline and publishes the resulting private dataset to Hugging Face. It supports the original three-zone run and a Nigeria-wide state/FCT run using 37 representative state coordinates.

## Required GitHub Actions secrets

- `HF_TOKEN`: Hugging Face token with write access.
- `EARTHDATA_TOKEN`: NASA Earthdata token.

Add them under **Settings → Secrets and variables → Actions**. Never commit tokens to this repository.

## Running

The workflow supports manual dispatch with `test_mode=true` by default. Choose `three_zones` to use Kano, Katsina, and Sokoto, or `nigeria_states` to use all 36 states plus the Federal Capital Territory. The national run uploads to `rufatronics/fieldshift-ng-data-nigeria`; the three-zone production run uploads to `rufatronics/fieldshift-ng-data`. The scheduled run uses the full three-zone configuration.

The national continuation workflow runs in resumable chunks: each chunk has a 160-minute data budget inside a 180-minute GitHub Actions timeout, saves its per-state/year cache even when it pauses, and runs again every four hours (approximately three hours of work followed by one hour of gap). POWER and SRTM state work is parallelized; IMERG years and MODIS composites resume from completed checkpoints. A completion marker prevents a later scheduled run from reprocessing an already-uploaded national dataset.

## Accessing zone data

The main three-zone dataset is `rufatronics/fieldshift-ng-data`. It contains 10-day dekads for Kano, Katsina, and Sokoto from 2005–2025. Read `manifest.json` first, then use `dekad_harmonized.parquet` for the wide table or `dekad_long_tagged.parquet` for source/unit-tagged values. The dataset can be downloaded with:

```python
import os, pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="rufatronics/fieldshift-ng-data",
    repo_type="dataset",
    filename="dekad_harmonized.parquet",
    token=os.environ["HF_TOKEN"],
)
df = pd.read_parquet(path)
katsina = df[df["zone"] == "katsina"]
```

The same table includes fields such as `power_precip_sum_mm`, `imerg_precip_sum_mm`, `modis_ndvi_mean`, temperature, humidity, solar radiation, soil wetness, and dekad metadata. `terrain.json` contains SRTM elevation, slope, relief, and relative position for each zone. Use a Hugging Face token with read access; never commit it to the repository.
