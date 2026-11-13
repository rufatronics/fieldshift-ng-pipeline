#!/usr/bin/env python3
"""
FieldShift NG - NASA data pipeline (northern Nigeria)  [consolidated v3]
=====================================================================
Pulls NASA POWER, GPM IMERG, MODIS NDVI (ORNL DAAC) and SRTM terrain for a few zones,
harmonizes everything to one time step (10-day dekads) and one analysis unit (zone),
tags every value with source + unit, and pushes a PRIVATE dataset to Hugging Face.

Run:
    python fieldshift_pipeline.py

Secrets (never hardcode them):
    EARTHDATA_TOKEN  -> https://urs.earthdata.nasa.gov  (Profile > Generate Token)
    HF_TOKEN         -> https://huggingface.co/settings/tokens  (write access)
Colab: put both in the Secrets panel (key icon) and enable notebook access.
Local: export them, or put them in a .env file next to this script, or type them
       at the prompt when asked.

First run: leave TEST_MODE = True (1 zone, 2 years, uploads to a "-test" repo).
When every stage says ok, set TEST_MODE = False and run again.

Before the full run, if you already have a local cache in Colab, move it to Drive:
    from google.colab import drive; drive.mount("/content/drive")
    !mkdir -p /content/drive/MyDrive/fieldshift_cache
    !cp -rn /content/fieldshift_cache/* /content/drive/MyDrive/fieldshift_cache/
then set USE_DRIVE_CACHE = True.
"""

# ============================================================
# 0) Dependencies (auto-install what is missing)
# ============================================================
import importlib
import subprocess
import sys


def ensure_packages(pkgs: dict) -> None:
    """pkgs = {import_name: pip_name}. Installs only what is missing."""
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("installing missing packages:", ", ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()


ensure_packages({
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "pyarrow": "pyarrow",
    "xarray": "xarray",
    "h5netcdf": "h5netcdf",
    "earthaccess": "earthaccess",
    "huggingface_hub": "huggingface_hub",
})

import datetime as dt
import getpass
import json
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================
# 1) CONFIG
# ============================================================
def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


TEST_MODE = env_bool("FIELD_SHIFT_TEST_MODE", True)
START_YEAR = int(os.getenv("FIELD_SHIFT_START_YEAR", "2005"))
END_YEAR = int(os.getenv("FIELD_SHIFT_END_YEAR", "2025"))
IMERG_START_YEAR = int(os.getenv("FIELD_SHIFT_IMERG_START_YEAR", "2015"))
USE_IMERG = env_bool("FIELD_SHIFT_USE_IMERG", True)
USE_MODIS = env_bool("FIELD_SHIFT_USE_MODIS", True)
USE_SRTM = env_bool("FIELD_SHIFT_USE_SRTM", True)
UPLOAD_TO_HF = env_bool("FIELD_SHIFT_UPLOAD_TO_HF", True)

HF_OWNER = os.getenv("FIELD_SHIFT_HF_OWNER") or None
HF_REPO_NAME = os.getenv("FIELD_SHIFT_HF_REPO_NAME", "fieldshift-ng-data")
HF_PRIVATE = env_bool("FIELD_SHIFT_HF_PRIVATE", True)
USE_DRIVE_CACHE = False      # SET True for the full run (Colab): finished years survive a disconnect
PIPELINE_SCOPE = os.getenv("FIELD_SHIFT_SCOPE", "three_zones")

ZONES = {
    "kano":    dict(name="Kano",    lat=12.0022, lon=8.5919),
    "katsina": dict(name="Katsina", lat=12.9908, lon=7.6018),
    "sokoto":  dict(name="Sokoto",  lat=13.0059, lon=5.2476),
}
if PIPELINE_SCOPE == "nigeria_states":
    state_file = Path(__file__).resolve().parent / "nigeria_states.json"
    ZONES = json.loads(state_file.read_text())
elif PIPELINE_SCOPE != "three_zones":
    raise ValueError(f"Unsupported FIELD_SHIFT_SCOPE: {PIPELINE_SCOPE}")
BOX_DEG = 0.25               # IMERG: mean over zone point +/- this many degrees
MODIS_KM = 1                 # MODIS: pixels within +/- this many km of the point
TERRAIN_BOX_DEG = 0.05       # SRTM: terrain window half-width (~5.5 km)
HEAT_DAY_C, RAIN_DAY_MM, HEAVY_RAIN_MM = 35.0, 1.0, 25.0
MIN_COVERAGE = 0.7           # a dekad needs >= 70% valid days, otherwise it stays missing
WET_SEASON_MONTHS = (6, 7, 8, 9)   # Jun-Sep, for the wet-season-only rainfall comparison

# "download" = bulk-download a year in parallel, then read locally (fast outside AWS).
# "stream"   = read remote granules directly (only fast inside AWS us-west-2).
# "auto"     = try stream, fall back to download.
IMERG_ACCESS = "download"
IMERG_WORKERS = 8            # parallel granule reads
IMERG_DOWNLOAD_THREADS = 8
KEEP_IMERG_RAW = False       # False = delete raw granules after extraction (saves GBs)
IMERG_YEAR_MIN_VALID = 0.8   # do not cache a year with more than 20% failed granules
# GPM_3IMERGDF V07 stores DAILY TOTALS in mm/day already (the producer does mean mm/hr x 24).
# Leave 1.0. Only change if you switch to a rate product.
IMERG_RATE_TO_DAY = 1.0

MODIS_CHUNK = 8              # composites per /subset call; the API hard limit is 10

if TEST_MODE:
    ZONES = dict(list(ZONES.items())[:1])
    START_YEAR, END_YEAR, IMERG_START_YEAR = 2023, 2024, 2024
    HF_REPO_NAME += "-test"

# ============================================================
# 2) Paths, secrets, HTTP helper
# ============================================================
IN_COLAB = Path("/content").is_dir()
BASE = Path("/content") if IN_COLAB else Path.cwd()
OUT = BASE / "fieldshift_data"

if USE_DRIVE_CACHE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = Path("/content/drive/MyDrive/fieldshift_cache")
else:
    if USE_DRIVE_CACHE:
        print("USE_DRIVE_CACHE ignored (not in Colab) -> local cache")
    CACHE = BASE / "fieldshift_cache"
CACHE.mkdir(parents=True, exist_ok=True)

STATUS = {}      # stage -> "ok" / "partial: ..." / "failed: ..." / "skipped"
DETAILS = {}     # extra per-stage diagnostics for the manifest


def _load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def get_secret(name):
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    v = os.environ.get(name)
    if v:
        return v
    if not IN_COLAB and not sys.stdin.isatty():
        raise RuntimeError(f"{name} is not set (Colab Secrets, environment, or .env)")
    return getpass.getpass(f"{name}: ").strip()


def get_json(url, params=None, tries=4, timeout=180):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(3 * 2 ** i)
    raise last


print("zones:", list(ZONES), "| years:", START_YEAR, "-", END_YEAR,
      "| test mode:", TEST_MODE, "| imerg access:", IMERG_ACCESS, "| cache:", CACHE, flush=True)

# ============================================================
# 3) Common time step: dekads (1-10, 11-20, 21-end of month)
# ============================================================
def add_dekad(idx):
    idx = pd.DatetimeIndex(idx)
    d = np.asarray(idx.day)
    part = np.where(d <= 10, 1, np.where(d <= 20, 2, 3))
    start = pd.to_datetime(pd.DataFrame({"year": np.asarray(idx.year),
                                         "month": np.asarray(idx.month),
                                         "day": (part - 1) * 10 + 1}))
    return start.values, (np.asarray(idx.month) - 1) * 3 + part   # start date, dekad-of-year 1..36


ALL_DEKADS = pd.DatetimeIndex(
    np.unique(add_dekad(pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))[0]))


def max_run(vals, thr):
    best = run = 0
    for v in vals:
        if v == v and v < thr:      # NaN breaks the run
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def agg(s, fn):
    """daily Series -> dekad Series. Missing days are NOT filled:
    a dekad with < MIN_COVERAGE valid days stays NaN."""
    s = s.sort_index()
    if s.index.has_duplicates:
        s = s.groupby(level=0).mean()
    s = s.asfreq("D")
    if s.empty:
        return pd.Series(dtype=float)
    key = pd.Series(add_dekad(s.index)[0], index=s.index)
    out = {k: (fn(g) if g.notna().mean() >= MIN_COVERAGE else np.nan)
           for k, g in s.groupby(key)}
    return pd.Series(out, dtype=float)


f_mean = lambda g: g.mean()
f_sum = lambda g: g.mean() * len(g)          # scaled to the full dekad length
f_max = lambda g: g.max()
f_min = lambda g: g.min()
f_days = lambda thr: (lambda g: (g.dropna() >= thr).mean() * len(g))
f_dry = lambda g: float(max_run(g.values, RAIN_DAY_MM))


def rain_block(p, prefix):
    return {f"{prefix}_precip_sum_mm": agg(p, f_sum),
            f"{prefix}_rain_days": agg(p, f_days(RAIN_DAY_MM)),
            f"{prefix}_heavy_rain_days": agg(p, f_days(HEAVY_RAIN_MM)),
            f"{prefix}_max_daily_mm": agg(p, f_max),
            f"{prefix}_max_dry_spell_days": agg(p, f_dry)}


def modis_dekad(df):
    s = df.set_index("date")["ndvi"].sort_index()
    s = s[(s >= -0.2) & (s <= 1.0)]
    if s.empty:
        return {}
    centre = s.index + pd.Timedelta(days=8)                     # real composite centre
    obs = pd.Series(1.0, index=centre).resample("D").sum()
    daily = s.copy()
    daily.index = centre
    daily = daily.resample("D").mean().interpolate("time", limit=32, limit_area="inside")
    return {"modis_ndvi_mean": agg(daily, f_mean),
            "modis_ndvi_min": agg(daily, f_min),
            "modis_ndvi_has_obs": agg(obs, lambda g: float(g.sum() > 0))}

# ============================================================
# 4) NASA POWER (no login)
# ============================================================
POWER_CORE = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
POWER_EXTRA = ["GWETROOT", "GWETTOP"]   # MERRA-2 soil wetness; dropped if the API rejects them
POWER_UNITS = {}
RENAME = {"T2M": "tmean", "T2M_MAX": "tmax", "T2M_MIN": "tmin", "RH2M": "rh",
          "ALLSKY_SFC_SW_DWN": "solar", "PRECTOTCORR": "precip",
          "GWETROOT": "wet_root", "GWETTOP": "wet_top"}


def fetch_power(zid, z):
    f = CACHE / f"power_{zid}_{START_YEAR}_{END_YEAR}.parquet"
    fu = f.with_suffix(".units.json")
    if f.exists() and fu.exists():
        POWER_UNITS.update(json.loads(fu.read_text()))
        return pd.read_parquet(f)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    js, err = None, None
    for plist in (POWER_CORE + POWER_EXTRA, POWER_CORE):
        try:
            js = get_json(url, dict(parameters=",".join(plist), community="AG",
                                    longitude=z["lon"], latitude=z["lat"],
                                    start=f"{START_YEAR}0101", end=f"{END_YEAR}1231",
                                    format="JSON"))
            break
        except Exception as e:
            err = e
    if js is None:
        raise err
    df = pd.DataFrame(js["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, np.nan).rename(columns=RENAME)
    units = {RENAME.get(k, k): v.get("units")
             for k, v in js.get("parameters", {}).items() if isinstance(v, dict)}
    POWER_UNITS.update(units)
    df.to_parquet(f)
    fu.write_text(json.dumps(units))
    return df

# ============================================================
# 5) GPM IMERG daily (Earthdata token)
# ============================================================
_EA = None
IMERG_UNITS = None


def earthdata_login():
    global _EA
    if _EA is not None:
        return _EA
    import earthaccess
    try:
        os.environ["EARTHDATA_TOKEN"] = get_secret("EARTHDATA_TOKEN")
    except Exception as e:
        print("no Earthdata token available:", e)
    auth = None
    for strategy in ("environment", "netrc", "interactive"):
        try:
            auth = earthaccess.login(strategy=strategy)
        except Exception as e:
            print(f"  earthaccess login [{strategy}] failed: {e!r}")
            continue
        if auth is not None and getattr(auth, "authenticated", False):
            print("Earthdata login ok via", strategy)
            _EA = earthaccess
            return _EA
    raise RuntimeError("Earthdata login failed (set EARTHDATA_TOKEN)")


def granule_date(g):
    try:
        return pd.Timestamp(g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10])
    except Exception:
        pass
    txt = str(g.get("meta", {}).get("native-id", ""))
    m = re.search(r"3IMERG\.(\d{8})", txt)
    if m:
        return pd.Timestamp(m.group(1))
    raise KeyError("cannot read granule date")


def granule_name(g):
    n = g.get("meta", {}).get("native-id")
    if n:
        return str(n)
    try:
        return g.data_links(access="external")[0].rsplit("/", 1)[-1]
    except Exception:
        return ""


def box_mean(p):
    """IMERG 'precipitation' DataArray -> ({zone: mm/day}, units_string).

    V07 files store dims as (time, lon, lat). Label-based .sel() copes with that,
    but we normalise to (..., lat, lon) with ascending coordinates anyway: it makes
    the box order-independent and safe if a future version flips lat direction.
    The sub.size check is what actually prevents a silent all-NaN box.
    """
    units = str(p.attrs.get("units", "unknown"))
    if "lat" in p.dims and "lon" in p.dims:
        p = p.transpose(..., "lat", "lon").sortby("lat").sortby("lon")
    row = {}
    for zid, z in ZONES.items():
        sub = p.sel(lon=slice(z["lon"] - BOX_DEG, z["lon"] + BOX_DEG),
                    lat=slice(z["lat"] - BOX_DEG, z["lat"] + BOX_DEG))
        if sub.size == 0:
            row[zid] = np.nan
            continue
        sub = sub.where(np.isfinite(sub) & (sub >= 0))
        v = float(sub.mean(skipna=True))
        row[zid] = v * IMERG_RATE_TO_DAY if np.isfinite(v) else np.nan
    return row, units


def _read_precip(fileobj):
    """Open one IMERG daily granule (remote file-like or local path) and box-average it."""
    import xarray as xr
    with xr.open_dataset(fileobj, engine="h5netcdf") as ds:
        if "precipitation" in ds.variables:
            return box_mean(ds["precipitation"])
    # half-hourly / HDF5-style granules keep everything under a "Grid" group
    with xr.open_dataset(fileobj, engine="h5netcdf", group="Grid") as ds:
        return box_mean(ds["precipitation"])


def _pairs_stream(ea, res):
    files = ea.open(res)
    if not files or len(files) != len(res):
        raise RuntimeError(f"streaming returned {0 if not files else len(files)}/{len(res)} handles")
    pairs = [(granule_date(g), f) for g, f in zip(res, files)]
    _read_precip(pairs[0][1])          # probe: fails fast if out-of-region
    return pairs


def _pairs_download(ea, res, year):
    dest = CACHE / "imerg_raw" / str(year)
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in ea.download(res, str(dest), threads=IMERG_DOWNLOAD_THREADS) if p]
    by_name = {p.name: p for p in paths}
    pairs = []
    for g in res:
        d = granule_date(g)
        p = by_name.get(granule_name(g))
        if p is None:
            p = next((q for q in paths if d.strftime("%Y%m%d") in q.name), None)
        if p is not None:
            pairs.append((d, str(p)))
    if not pairs:
        raise RuntimeError("download returned no usable files")
    return pairs


def fetch_imerg_year(ea, year):
    global IMERG_UNITS
    f = CACHE / f"imerg_{year}_box{BOX_DEG}_{'_'.join(ZONES)}.parquet"
    fu = f.with_suffix(".units.json")
    if f.exists():
        if fu.exists():
            IMERG_UNITS = json.loads(fu.read_text()).get("units", IMERG_UNITS)
        return pd.read_parquet(f)

    res = ea.search_data(short_name="GPM_3IMERGDF", version="07",
                         temporal=(f"{year}-01-01", f"{year}-12-31"), count=400)
    if not res:
        print(f"  IMERG {year}: no granules found")
        return pd.DataFrame(columns=list(ZONES), dtype=float)
    res = sorted(res, key=granule_date)

    mode_used, pairs, last = None, None, None
    order = {"auto": ("stream", "download"), "stream": ("stream",), "download": ("download",)}
    for mode in order[IMERG_ACCESS]:
        try:
            pairs = _pairs_stream(ea, res) if mode == "stream" else _pairs_download(ea, res, year)
            mode_used = mode
            break
        except Exception as e:
            last = e
            print(f"  IMERG {year}: {mode} access failed ({e!r})")
    if pairs is None:
        raise RuntimeError(f"no IMERG access method worked: {last!r}")
    print(f"  IMERG {year}: {mode_used} mode, {len(pairs)}/{len(res)} granules available", flush=True)

    def one(pair):
        date, fobj = pair
        for _ in range(3):
            try:
                row, units = _read_precip(fobj)
                return date, row, units
            except Exception:
                time.sleep(2)
        return date, {zid: np.nan for zid in ZONES}, None

    with ThreadPoolExecutor(max_workers=IMERG_WORKERS) as ex:
        rows = list(ex.map(one, pairs))
    for _, _, u in rows:
        if u:
            IMERG_UNITS = u
            break

    df = pd.DataFrame({d: r for d, r, _ in rows}).T.sort_index()
    df = df[~df.index.duplicated()].astype(float)
    valid = float(df.notna().mean().min()) if len(df) else 0.0
    if len(df) and valid >= IMERG_YEAR_MIN_VALID:
        df.to_parquet(f)
        fu.write_text(json.dumps({"units": IMERG_UNITS, "access": mode_used}))
    else:
        print(f"  warning: IMERG {year} only {valid:.0%} valid - not cached, rerun to retry")

    if mode_used == "download" and not KEEP_IMERG_RAW:
        shutil.rmtree(CACHE / "imerg_raw" / str(year), ignore_errors=True)
    DETAILS.setdefault("imerg_years", {})[str(year)] = dict(
        granules=len(res), read=len(df), valid_fraction=round(valid, 3), access=mode_used)
    return df


def run_imerg():
    if not USE_IMERG:
        STATUS["imerg"] = "skipped"
        return {}
    try:
        ea = earthdata_login()
        frames = []
        for y in range(max(IMERG_START_YEAR, START_YEAR), END_YEAR + 1):
            t0 = time.time()
            frames.append(fetch_imerg_year(ea, y))
            print(f"IMERG {y} done in {time.time() - t0:.0f}s", flush=True)
        frames = [f for f in frames if len(f)]
        if not frames:
            raise RuntimeError("no IMERG data returned")
        im = pd.concat(frames)
        im.index = pd.DatetimeIndex(im.index)
        DETAILS["imerg_native_units_attribute"] = IMERG_UNITS
        DETAILS["imerg_rate_to_day_factor"] = IMERG_RATE_TO_DAY
        STATUS["imerg"] = "ok"
        return {zid: im[zid].astype(float) for zid in ZONES if zid in im}
    except Exception as e:
        STATUS["imerg"] = f"failed: {e!r}"
        print("IMERG skipped ->", STATUS["imerg"], "(POWER precipitation still carries the dataset)")
        return {}

# ============================================================
# 6) MODIS NDVI via ORNL DAAC TESViS (no login)
#    The API allows at most 10 MODIS dates per /subset call and it
#    intermittently returns 502/503. Strategy: small chunks that never
#    cross a calendar year, halve the chunk on failure down to single
#    dates, and checkpoint after every chunk so a crash loses nothing.
# ============================================================
ORNL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"


def _modis_date_to_ts(md):
    return pd.Timestamp(year=int(md[1:5]), month=1, day=1) + pd.Timedelta(days=int(md[5:]) - 1)


def _ornl_rows(z, chunk):
    """One /subset call. On failure, split the chunk and retry down to single dates."""
    try:
        js = get_json(f"{ORNL}/subset",
                      dict(latitude=z["lat"], longitude=z["lon"],
                           startDate=chunk[0]["modis_date"], endDate=chunk[-1]["modis_date"],
                           kmAboveBelow=MODIS_KM, kmLeftRight=MODIS_KM),
                      tries=3, timeout=120)
    except Exception as e:
        if len(chunk) == 1:
            print(f"  MODIS: skipping {chunk[0]['modis_date']} ({type(e).__name__}) - will retry next run")
            return []
        mid = len(chunk) // 2
        print(f"  MODIS: {len(chunk)}-date chunk failed ({type(e).__name__}), splitting into {mid}+{len(chunk)-mid}")
        time.sleep(5)
        return _ornl_rows(z, chunk[:mid]) + _ornl_rows(z, chunk[mid:])

    by = {}
    for it in js.get("subset", []):
        key = "ndvi" if "NDVI" in it["band"] else ("rel" if "reliab" in it["band"].lower() else None)
        if key:
            by.setdefault(it["modis_date"], {})[key] = np.array(it["data"], dtype=float)
            by[it["modis_date"]]["calendar_date"] = it.get("calendar_date")
    rows = []
    for md, v in by.items():
        if "ndvi" not in v:
            continue
        nd = v["ndvi"]
        raw = np.nanmax(np.abs(nd)) > 2                 # raw integers (x10000) or already scaled
        ok = nd > (-2999 if raw else -0.2999)
        if "rel" in v:
            ok &= np.isin(v["rel"], [0, 1])             # good + marginal pixels only
        val = float(nd[ok].mean() * (1e-4 if raw else 1.0)) if ok.any() else np.nan
        cd = v.get("calendar_date") or _modis_date_to_ts(md).strftime("%Y-%m-%d")
        rows.append((cd, val, int(ok.sum()), int(nd.size), md))
    return rows


def fetch_modis(zid, z):
    f = CACHE / f"modis_{zid}_{START_YEAR}_{END_YEAR}_km{MODIS_KM}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    part = f.with_suffix(".partial.json")
    done = json.loads(part.read_text()) if part.exists() else {}
    if done:
        print(f"  MODIS {zid}: resuming with {len(done)} composites already fetched")

    dj = CACHE / f"modis_dates_{zid}.json"
    if dj.exists():
        dates = json.loads(dj.read_text())
    else:
        dates = get_json(f"{ORNL}/dates", dict(latitude=z["lat"], longitude=z["lon"]))["dates"]
        dj.write_text(json.dumps(dates))
    dates = [d for d in dates if START_YEAR <= int(d["calendar_date"][:4]) <= END_YEAR]
    todo = [d for d in dates if d["modis_date"] not in done]

    for yr in sorted({d["calendar_date"][:4] for d in todo}):
        yr_dates = [d for d in todo if d["calendar_date"][:4] == yr]
        for i in range(0, len(yr_dates), MODIS_CHUNK):
            ch = yr_dates[i:i + MODIS_CHUNK]
            for cd, val, nok, npx, md in _ornl_rows(z, ch):
                done[md] = [cd, val, nok, npx]
            part.write_text(json.dumps(done))           # checkpoint after every chunk
            time.sleep(0.5)

    rows = [(pd.Timestamp(v[0]), v[1], v[2], v[3]) for v in done.values()]
    df = pd.DataFrame(rows, columns=["date", "ndvi", "n_ok", "n_px"]).sort_values("date").reset_index(drop=True)
    missing = len(dates) - len(df)
    DETAILS.setdefault("modis_missing_composites", {})[zid] = int(missing)
    if missing:
        print(f"  MODIS {zid}: {missing} composites still missing - rerun to retry just those")
    if len(df) and not missing:
        df.to_parquet(f)                                # only a COMPLETE series becomes the final cache
        part.unlink(missing_ok=True)
    return df                                           # incomplete -> next run retries just the gaps


def run_modis():
    if not USE_MODIS:
        STATUS["modis"] = "skipped"
        return {}
    out, errs = {}, {}
    for zid, z in ZONES.items():
        try:
            out[zid] = fetch_modis(zid, z)
            print("MODIS", zid, out[zid].shape, "valid:", int(out[zid].ndvi.notna().sum()), flush=True)
        except Exception as e:
            errs[zid] = repr(e)
            print("MODIS failed for", zid, "->", repr(e))
    miss = DETAILS.get("modis_missing_composites", {})
    if errs:
        STATUS["modis"] = f"partial: {errs}" if out else f"failed: {errs}"
    elif any(miss.values()):
        STATUS["modis"] = f"partial: missing composites {miss}"
    else:
        STATUS["modis"] = "ok"
    return out

# ============================================================
# 7) SRTM terrain (Earthdata token)
# ============================================================
def tile_name(lat, lon):
    return (f"{'N' if lat >= 0 else 'S'}{abs(int(np.floor(lat))):02d}"
            f"{'E' if lon >= 0 else 'W'}{abs(int(np.floor(lon))):03d}")


def _read_hgt(path):
    path = str(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            inner = next(n for n in zf.namelist() if n.lower().endswith(".hgt"))
            buf = zf.read(inner)
    else:
        buf = Path(path).read_bytes()
    arr = np.frombuffer(buf, dtype=">i2")
    n = int(round(np.sqrt(arr.size)))
    if n * n != arr.size:
        raise ValueError(f"unexpected SRTM tile size: {arr.size}")
    a = arr.reshape(n, n).astype(float)
    a[a == -32768] = np.nan
    return a


def terrain_stats(a, lat, lon):
    n = a.shape[0]
    step = n - 1                                        # cells per degree
    r = int(round((np.floor(lat) + 1 - lat) * step))
    c = int(round((lon - np.floor(lon)) * step))
    hw = max(int(TERRAIN_BOX_DEG * step), 1)
    w = a[max(r - hw, 0): r + hw + 1, max(c - hw, 0): c + hw + 1]
    vals = w[np.isfinite(w)]
    if vals.size < 9:
        raise ValueError("SRTM window is mostly void")
    dy = 111320.0 / step
    dx = dy * np.cos(np.radians(lat))
    gy, gx = np.gradient(np.nan_to_num(w, nan=float(np.nanmean(w))), dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    elev = float(np.nanmean(a[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2]))
    return dict(tile=tile_name(lat, lon), elev_m=round(elev, 1),
                slope_deg_mean=round(float(np.nanmean(slope)), 2),
                slope_deg_p90=round(float(np.nanpercentile(slope, 90)), 2),
                relief_m=round(float(vals.max() - vals.min()), 1),
                relative_position=round(float((vals <= elev).mean()), 3),  # 0 = lowest ground (flood-prone)
                window_km=round(2 * TERRAIN_BOX_DEG * 111.32, 1))


def terrain_for(ea, z):
    zid = z["_id"]
    f = CACHE / f"srtm_{zid}_box{TERRAIN_BOX_DEG}.json"
    if f.exists():
        return json.loads(f.read_text())
    name = tile_name(z["lat"], z["lon"])
    res = ea.search_data(short_name="SRTMGL1", version="003",
                         bounding_box=(z["lon"] - 0.01, z["lat"] - 0.01,
                                       z["lon"] + 0.01, z["lat"] + 0.01))
    if not res:
        raise RuntimeError(f"no SRTMGL1 granule found for tile {name}")
    paths = [str(p) for p in ea.download(res, str(CACHE / "srtm")) if p]
    zp = next((p for p in paths if name in Path(p).name), None)
    if zp is None:
        raise RuntimeError(f"tile {name} not in downloads: {[Path(p).name for p in paths]}")
    out = terrain_stats(_read_hgt(zp), z["lat"], z["lon"])
    f.write_text(json.dumps(out))
    return out


def run_srtm():
    if not USE_SRTM:
        STATUS["srtm"] = "skipped"
        return {}
    out, errs = {}, {}
    try:
        ea = earthdata_login()
    except Exception as e:
        STATUS["srtm"] = f"failed: {e!r}"
        print("SRTM skipped ->", STATUS["srtm"])
        return {}
    for zid, z in ZONES.items():
        try:
            out[zid] = terrain_for(ea, dict(z, _id=zid))
            print("SRTM", zid, out[zid], flush=True)
        except Exception as e:
            errs[zid] = repr(e)
            print("SRTM failed for", zid, "->", repr(e))
    STATUS["srtm"] = "ok" if out and not errs else (f"partial: {errs}" if out else f"failed: {errs}")
    return out

# ============================================================
# 8) Harmonize: one table, one time step, one analysis unit (zone)
# ============================================================
SRC = {
    "power": ("NASA POWER daily, AG community (MERRA-2/CERES based)",
              "about 0.5 x 0.625 deg cell at the zone point"),
    "imerg": ("NASA GPM IMERG Final Run V07 (GPM_3IMERGDF)",
              "0.1 deg, mean over the zone box"),
    "modis": ("MODIS Terra MOD13Q1 v061 via ORNL DAAC Web Service",
              "250 m, 16-day composites, mean over the zone window"),
}
INFO = {
    "t2m_mean_c": ("degC", "dekad mean of daily mean 2 m air temperature"),
    "t2m_max_mean_c": ("degC", "dekad mean of daily maximum temperature"),
    "t2m_min_mean_c": ("degC", "dekad mean of daily minimum temperature"),
    "heat_days": ("days", f"days per dekad with Tmax >= {HEAT_DAY_C} C (scaled to dekad length)"),
    "rh2m_mean_pct": ("%", "dekad mean relative humidity at 2 m"),
    "solar_mean": ("POWER units (see manifest.power_units)",
                   "dekad mean all-sky surface shortwave downward irradiance"),
    "soil_wetness_root": ("0-1", "MERRA-2 root-zone soil wetness (model-based, not satellite soil moisture)"),
    "soil_wetness_top": ("0-1", "MERRA-2 surface soil wetness (model-based, not satellite soil moisture)"),
    "precip_sum_mm": ("mm", "dekad rainfall total"),
    "rain_days": ("days", f"days with >= {RAIN_DAY_MM} mm"),
    "heavy_rain_days": ("days", f"days with >= {HEAVY_RAIN_MM} mm (zone-mean values; point extremes are under-represented)"),
    "max_daily_mm": ("mm/day", "largest daily rainfall in the dekad"),
    "max_dry_spell_days": ("days", f"longest run of days < {RAIN_DAY_MM} mm inside the dekad (runs are not carried across dekads)"),
    "ndvi_mean": ("NDVI", "dekad mean of linearly interpolated 16-day composites (reliability 0/1 pixels only)"),
    "ndvi_min": ("NDVI", "dekad minimum of the interpolated NDVI"),
    "ndvi_has_obs": ("0/1", "1 if a real composite centre falls inside the dekad, 0 = value is interpolated"),
}


def meta(col):
    p, k = col.split("_", 1)
    u, d = INFO[k]
    s, r = SRC[p]
    if col == "power_solar_mean":
        u = POWER_UNITS.get("solar") or u
    return dict(source=s, native_resolution=r, unit=u, description=d)


def harmonize(POWER, IMERG, MODIS):
    frames = []
    for zid in ZONES:
        if zid not in POWER:
            continue
        d, t = POWER[zid], {}
        t["power_t2m_mean_c"] = agg(d.tmean, f_mean)
        t["power_t2m_max_mean_c"] = agg(d.tmax, f_mean)
        t["power_t2m_min_mean_c"] = agg(d.tmin, f_mean)
        t["power_heat_days"] = agg(d.tmax, f_days(HEAT_DAY_C))
        t["power_rh2m_mean_pct"] = agg(d.rh, f_mean)
        t["power_solar_mean"] = agg(d.solar, f_mean)
        if "wet_root" in d:
            t["power_soil_wetness_root"] = agg(d.wet_root, f_mean)
        if "wet_top" in d:
            t["power_soil_wetness_top"] = agg(d.wet_top, f_mean)
        t.update(rain_block(d.precip, "power"))
        if zid in IMERG:
            t.update(rain_block(IMERG[zid], "imerg"))
        if zid in MODIS:
            t.update(modis_dekad(MODIS[zid]))
        w = pd.DataFrame(t).reindex(ALL_DEKADS)
        w.index.name = "dekad_start"
        w = w.reset_index()
        w.insert(0, "zone", zid)
        w.insert(2, "year", w.dekad_start.dt.year)
        w.insert(3, "dekad_of_year", add_dekad(pd.DatetimeIndex(w.dekad_start))[1])
        frames.append(w)
    if not frames:
        raise RuntimeError("nothing to harmonize: POWER returned no zones")
    return pd.concat(frames, ignore_index=True)


def rainfall_validation(wide):
    """POWER vs IMERG at three time scales. Report the scale you make decisions at:
    totals can agree while dekad timing does not."""
    valid = {}
    if "imerg_precip_sum_mm" not in wide:
        return valid
    for zid, g in wide.groupby("zone"):
        s = g[["dekad_start", "power_precip_sum_mm", "imerg_precip_sum_mm"]].dropna()
        if len(s) <= 10:
            continue
        ratio = float(s.imerg_precip_sum_mm.sum() / s.power_precip_sum_mm.sum())
        monthly = s.set_index("dekad_start").resample("MS").sum()   # dekads never straddle months
        wet = s[s.dekad_start.dt.month.isin(WET_SEASON_MONTHS)]
        valid[zid] = dict(
            n_dekads=int(len(s)),
            pearson_r_dekad=round(float(s.power_precip_sum_mm.corr(s.imerg_precip_sum_mm)), 3),
            pearson_r_monthly=round(float(monthly.power_precip_sum_mm.corr(monthly.imerg_precip_sum_mm)), 3),
            n_wet_dekads=int(len(wet)),
            pearson_r_wet_season=(round(float(wet.power_precip_sum_mm.corr(wet.imerg_precip_sum_mm)), 3)
                                  if len(wet) > 3 else None),
            total_power_mm=round(float(s.power_precip_sum_mm.sum()), 1),
            total_imerg_mm=round(float(s.imerg_precip_sum_mm.sum()), 1),
            imerg_to_power_total_ratio=round(ratio, 3),
            bias_pct=round(100 * (ratio - 1), 1))
        if not 0.3 <= ratio <= 3.0:
            print(f"  WARNING {zid}: IMERG/POWER rainfall ratio {ratio:.3f} is far from 1 "
                  f"- check IMERG_RATE_TO_DAY and the box size before trusting these numbers")
    return valid


def tag_and_summarize(wide):
    keys = ["zone", "dekad_start", "year", "dekad_of_year"]
    var_cols = [c for c in wide.columns if c not in keys]
    var_meta = {c: meta(c) for c in var_cols}

    long = wide.melt(id_vars=keys, var_name="variable", value_name="value").dropna(subset=["value"])
    long = long.join(pd.DataFrame(var_meta).T[["source", "unit"]], on="variable")

    clim = (long.groupby(["zone", "dekad_of_year", "variable"])["value"]
            .agg(mean="mean", std="std", p10=lambda s: s.quantile(0.1), p50="median",
                 p90=lambda s: s.quantile(0.9), n_years="count").reset_index())

    valid = rainfall_validation(wide)
    cover = {zid: wide[wide.zone == zid][var_cols].notna().mean().round(3).to_dict()
             for zid in wide.zone.unique()}
    return long, clim, var_cols, var_meta, valid, cover

# ============================================================
# 9) Save + manifest + README
# ============================================================
def save_all(wide, long, clim, var_cols, var_meta, valid, cover, terrain):
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    wide.to_parquet(OUT / "dekad_harmonized.parquet", index=False)
    wide.to_csv(OUT / "dekad_harmonized.csv", index=False)
    long.to_parquet(OUT / "dekad_long_tagged.parquet", index=False)
    clim.to_parquet(OUT / "climatology_dekad.parquet", index=False)
    (OUT / "zones.json").write_text(json.dumps(ZONES, indent=2))
    (OUT / "terrain.json").write_text(json.dumps(terrain, indent=2))

    manifest = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        test_mode=TEST_MODE, years=[START_YEAR, END_YEAR],
        imerg_start_year=max(IMERG_START_YEAR, START_YEAR),
        analysis_unit="zone (point or small box around the zone centre)",
        time_step="dekad (1-10, 11-20, 21-end of month)",
        thresholds=dict(heat_day_c=HEAT_DAY_C, rain_day_mm=RAIN_DAY_MM,
                        heavy_rain_mm=HEAVY_RAIN_MM, min_coverage=MIN_COVERAGE,
                        wet_season_months=list(WET_SEASON_MONTHS)),
        spatial=dict(imerg_box_deg=BOX_DEG, modis_km=MODIS_KM, terrain_box_deg=TERRAIN_BOX_DEG),
        stages=STATUS, stage_details=DETAILS,
        power_units=POWER_UNITS, imerg_units_attribute=IMERG_UNITS,
        variables=var_meta, validation=valid, coverage=cover,
        limitations=[
            "Zone-level analysis, not field-level: POWER cells are tens of km wide, IMERG is 0.1 deg, MODIS is a ~2 km window mean.",
            "POWER rainfall and soil wetness are reanalysis/model-based; IMERG is a satellite estimate. Neither is ground truth.",
            "Rainfall sources can agree on seasonal totals while disagreeing on dekad timing; see validation (r_dekad vs r_monthly).",
            "Dekads with under 70% valid days are left missing. Nothing is gap-filled except linear NDVI interpolation between 16-day composites (see ndvi_has_obs).",
            "Heavy-rain day counts from zone-mean values under-represent point extremes.",
            "Terrain comes from one SRTM 1 arc-second tile and a window of about 11 km.",
            "IMERG Final is delivered in monthly batches with roughly 3.5-4 months latency; the most recent months may be absent.",
            "No soil, crop, price or market data are included; those must come from local sources and farmers.",
        ])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    readme = """---
license: other
pretty_name: FieldShift NG harmonized NASA climate data (northern Nigeria)
tags: [nasa, earth-observation, agriculture, nigeria, climate]
---
# FieldShift NG harmonized NASA data (private)

One row per zone and 10-day dekad (1-10, 11-20, 21-end of month).

- `dekad_harmonized.parquet` / `.csv`: wide table, one column per variable
- `dekad_long_tagged.parquet`: every value with its source and unit
- `climatology_dekad.parquet`: per-zone mean, std, p10, p50, p90 for each dekad of the year
- `terrain.json`: SRTM elevation, slope, relief and relative position per zone
- `manifest.json`: sources, units, thresholds, stage status, validation, coverage, limitations
- `zones.json`: zone coordinates

Sources: NASA POWER (AG), NASA GPM IMERG Final V07 (GPM_3IMERGDF), MODIS MOD13Q1 v061 via the
ORNL DAAC TESViS Web Service, NASA SRTMGL1 v003. Cite each dataset according to its own landing page.
Missing values stay missing. Read `manifest.json` before using the data.
"""
    (OUT / "README.md").write_text(readme)
    print(sorted(p.name for p in OUT.iterdir()))
    print(wide.groupby("zone")[var_cols].apply(lambda g: g.notna().mean()).round(2).T)

# ============================================================
# 10) Push to a PRIVATE Hugging Face dataset
# ============================================================
def upload():
    if not UPLOAD_TO_HF:
        print("UPLOAD_TO_HF = False -> files kept at", OUT)
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=get_secret("HF_TOKEN"))
        owner = HF_OWNER or api.whoami()["name"]
        repo_id = f"{owner}/{HF_REPO_NAME}"
        api.create_repo(repo_id, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
        if HF_PRIVATE:
            api.update_repo_settings(repo_id, repo_type="dataset", private=True)
        api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="dataset",
                          commit_message=f"FieldShift NG data {START_YEAR}-{END_YEAR} "
                                         f"({'test' if TEST_MODE else 'full'})")
        print("uploaded:", f"https://huggingface.co/datasets/{repo_id}",
              "(private)" if HF_PRIVATE else "(PUBLIC)")
    except Exception as e:
        print("Hugging Face upload failed:", repr(e))
        print("Your files are safe locally at:", OUT)


def run_pipeline():
    t0 = time.time()
    POWER = {}
    for zid, z in ZONES.items():
        POWER[zid] = fetch_power(zid, z)
        print("POWER", zid, POWER[zid].shape, sorted(POWER[zid].columns), flush=True)
    STATUS["power"] = "ok"

    IMERG = run_imerg()
    MODIS = run_modis()
    TERRAIN = run_srtm()

    wide = harmonize(POWER, IMERG, MODIS)
    long, clim, var_cols, var_meta, valid, cover = tag_and_summarize(wide)
    print(wide.shape, "| long:", long.shape, "| climatology:", clim.shape)
    print("validation (POWER vs IMERG rainfall):", json.dumps(valid, indent=2))
    save_all(wide, long, clim, var_cols, var_meta, valid, cover, TERRAIN)
    upload()

    print("\n--- stage summary ---")
    for k in ("power", "imerg", "modis", "srtm"):
        print(f"{k:6s}: {STATUS.get(k, 'not run')}")
    print(f"total time: {time.time() - t0:.0f}s | output: {OUT}")
    if TEST_MODE:
        print("TEST_MODE was on: set it to False and run again for the full dataset.")
    return wide


if __name__ == "__main__":
    run_pipeline()
