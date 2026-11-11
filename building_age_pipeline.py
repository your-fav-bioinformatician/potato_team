"""
Building-age mapping from Landsat time series — full pipeline in one file.

  raw data (parquet, folder or .zip)  ->  preprocessing (v3 rules + 3x3 neighbourhood channels)
  ->  Madrid-trained TempCNN backbone (ensemble)  ->  modelCreate(dataset, labels)  ->  predictions

Rule for the labelled pixels you provide (k = labels of the smallest class):
  k <= 20  : BARE BACKBONE  — the backbone as it is; each pixel gets the class of the closest
             class prototype (mean embedding of the labelled pixels of that class).
  k  > 20  : FULL PIPELINE  — AdaBN on the dataset's unlabelled pixels, few-shot fine-tuning with
             gradual unfreezing chosen by spatial-block cross-validation, prototypes + kNN, and the
             confidence rule (margin between the two closest prototypes; low-margin pixels decided
             by their spatial neighbours).
Nothing depends on which city it is: hemisphere from the Landsat WRS rows, centre given (or the
centroid of the oldest-class labels), channels and years matched to the backbone by name/year,
all thresholds and distances measured on the dataset itself.

Labelled pixels per class: the task declares n in N_VALUES = (5, 25, 50, 100, 200)
  -> n = 5 uses the bare backbone, n = 25..200 the full pipeline.

Run:
  # one model from your own labels
  python building_age_pipeline.py --data city.parquet|folder|archive.zip --labels labels.csv --out results/
      [--centre LAT,LON | --centre-xy X,Y] [--bounds 1960,1984,2004] [--backbone path/backbone.pt]
  # one model from n labelled pixels per class drawn from the dataset's own labels (fixed seed)
  python building_age_pipeline.py --data city.parquet --bounds B1,B2,B3 --n 25 --seed 0 --out results/
  # the task's evaluation: many random draws for every n in N_VALUES -> mean +/- std macro-F1, table, plot
  python building_age_pipeline.py --data city.parquet --bounds B1,B2,B3 --evaluate --draws 10 --seed 0 --out results/
  labels.csv: px_key, py_key and either age_class (integers) or year (+ --bounds)
Python:
  from building_age_pipeline import prepare, modelCreate
  city = prepare("city.parquet", backbone="backbone.pt", centre_latlon=(lat, lon))
  model = modelCreate(city, labels_df); model.save("results/")
Outputs: predictions.parquet (+ .csv), report.json, model.pt
Requires: numpy, pandas, pyarrow, torch, scikit-learn (+ matplotlib, optional: --evaluate plot)
Files: backbone.pt with backbone_meta.json in the same folder (default: model/ next to this script)
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_BACKBONE_PLACES = [HERE / "model" / "backbone.pt",                               # repository layout
                    HERE / "results" / "rebuild" / "ensemble" / "backbone.pt"]      # development layout
DEFAULT_BACKBONE = next((p for p in _BACKBONE_PLACES if p.exists()), _BACKBONE_PLACES[0])   # latest Kaggle build
BARE_MAX_K = 20                         # labels per class at or below which the bare backbone is used
N_VALUES = (5, 25, 50, 100, 200)        # labelled pixels per class declared by the task (Introduction, 1.5)

torch.set_num_threads(int(os.environ.get("N_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
SMOKE = os.environ.get("MS_SMOKE") == "1"                                       # tiny functional test only

# ══════════════════════════════════════════════════════════════════════════════
# 1. Constants of the preprocessing rules and of the backbone's training recipe
# ══════════════════════════════════════════════════════════════════════════════
BASE_BANDS   = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
INDEX_COLS   = ["NDVI", "NDBI", "UI", "MNDWI", "BSI"]
SENSORS      = ["LT04", "LT05", "LE07", "LC08", "LC09"]
EARLY_CUTOFF, LATE_START = 2003, 2004
RADSAT_BIT   = {"TM": [0, 1, 2, 3, 4, 6], "OLI": [1, 2, 3, 4, 5, 6]}   # USGS C2 L2 QA_RADSAT
MID_SUMMER_DOY, Z_ANOM, CEIL_MIN_COUNT, MIN_ACQ_OBS = 196, 3.0, 10, 100
W_MIN, TAU, TAU_HEAP, Z_HEAP = 0.2, 1.0, 5.0, 3.0
META_CHANNELS = ([f"obs_{b}" for b in BASE_BANDS] + ["doy"] + [f"sensor_{s}" for s in SENSORS]
                 + ["n_valid_slots", "anomalous_acq", "tier2"])
NEIGH_CHANNELS = [f"neigh_{b}" for b in BASE_BANDS]
STATIC_COLS  = ["coverage", "dist_centre_rel"]
NB3_FEATURES = ([f"{c}_mean" for c in BASE_BANDS] + [f"{c}_std" for c in BASE_BANDS]
                + [f"{c}_mean" for c in INDEX_COLS] + [f"{c}_std" for c in INDEX_COLS]
                + [f"{b}_early_mean" for b in BASE_BANDS] + [f"{b}_early_std" for b in BASE_BANDS]
                + [f"{b}_late_mean" for b in BASE_BANDS] + [f"{b}_late_std" for b in BASE_BANDS]
                + [f"d_{b}_mean" for b in BASE_BANDS] + [f"d_{b}_std" for b in BASE_BANDS]
                + ["has_early_data", "has_late_data"])
TAB_FEATURES = NB3_FEATURES + STATIC_COLS
WEIGHT_DECAY, EMB_DIM, TAU0, DROPOUT0 = 1e-4, 32, 0.1, 0.3
FIXED = dict(lr=1e-3, per_class=64, lr_patience=2, lr_factor=0.5, min_lr=1e-5, grad_clip=1.0,
             max_epochs=40, patience=5)
LRS, ETAS = (1e-4, 3e-4, 1e-3), (3, 10)        # fine-tuning grid (eta: LR / eta per lower layer)
WRS_EQUATOR_ROW = 60                            # Landsat WRS-2: daytime row 60 crosses the equator


def log(s):
    print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)


def compute_indices(B, G, R, N, S1, S2):
    eps = 1e-6
    return [(N - R) / (N + R + eps), (S1 - N) / (S1 + N + eps), (S2 - N) / (S2 + N + eps),
            (G - S1) / (G + S1 + eps), ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + eps)]


def utm(lat, lon, zone=None):
    """WGS84 -> UTM, standard transverse-Mercator series. Zone from the longitude if not given;
    southern hemisphere uses the 10,000 km false northing."""
    if zone is None:
        zone = int((lon + 180) // 6) + 1
    south = lat < 0
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f); ep2 = e2 / (1 - e2)
    lat, lon0 = np.radians(lat), np.radians((zone - 1) * 6 - 180 + 3)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2); T = np.tan(lat) ** 2; C = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (np.radians(lon) - lon0)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120)
    y = k0 * (M + N * np.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                     + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x + 500000, (y + 10_000_000 if south else y)

def label_weights(wmy, bounds):
    """Training weight = reliability of a label: heaped construction years (detected from the
    labels' own histogram) and closeness to a class boundary lower the weight (floor W_MIN)."""
    years = np.round(wmy).astype(int)
    is_int = np.isclose(wmy, years)
    counts = pd.Series(years[is_int]).value_counts()
    yr = np.arange(int(wmy.min()), int(wmy.max()) + 1)
    c = counts.reindex(yr, fill_value=0).to_numpy(float)
    ratio = np.ones_like(c)
    for i in range(len(yr)):
        nb = [c[j] for j in range(max(0, i - 3), min(len(yr), i + 4)) if j != i]
        ratio[i] = (c[i] + 1) / (np.median(nb) + 1)
    lr = np.log(ratio)
    mad = 1.4826 * np.median(np.abs(lr - np.median(lr)))
    heaped_years = yr[(lr - np.median(lr)) / mad > Z_HEAP] if mad > 0 else yr[:0]
    heaped = is_int & np.isin(years, heaped_years)
    d = np.abs(wmy[:, None] - np.asarray(bounds, float)[None, :]).min(axis=1)
    tau = np.where(heaped, TAU_HEAP, TAU)
    w = W_MIN + (1 - W_MIN) * (1 - np.exp(-d / tau))
    return w.astype(np.float32), dict(heaped_years=[int(y) for y in heaped_years])


# ══════════════════════════════════════════════════════════════════════════════
# 2. Preprocessing (v3 rules, any dataset in the hackathon parquet format)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_dataset(path, centre_xy=None, class_bounds=None, doy_shift=0.0, strict=True,
                       keep_xy=False, return_internals=False):
    """The v3 rules for ANY dataset in the hackathon parquet format. Nothing here depends on
    which city it is; everything city-dependent is an argument:
      centre_xy    centre in the dataset's own projected coordinates (None -> dist_centre_rel NaN)
      class_bounds the dataset's class boundary years (None or no label column -> no labels)
      doy_shift    days added to every acquisition DOY before the season rules (0 in the
                   northern hemisphere; half a year in the southern one, so 'summer' means summer)
      strict       False: a pixel-band with no usable observation in any year is filled with the
                   dataset's per-year median of observed values (still flagged unobserved)"""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    cols = ["year", "px_key", "py_key", "pixel_x", "pixel_y", "coverage"]
    has_labels = "weighted_mean_year" in have and class_bounds is not None
    if has_labels:
        cols.append("weighted_mean_year")
    for s in (1, 2, 3):
        cols += [f"{b}_{s}" for b in BASE_BANDS]
        cols += [f"QA_PIXEL_{s}", f"QA_RADSAT_{s}", f"doy_{s}", f"scene_id_{s}", f"qa_valid_{s}"]
    missing = [c for c in cols if c not in have]
    if missing:
        raise ValueError(f"dataset lacks required columns: {missing}")
    df = pd.read_parquet(path, columns=cols)
    key = df["px_key"].to_numpy(np.int64) * 1_000_000 + df["py_key"].to_numpy(np.int64)
    ukey, first, pix = np.unique(key, return_index=True, return_inverse=True)
    y0 = int(df["year"].min()); years = np.arange(y0, int(df["year"].max()) + 1)
    t = df["year"].to_numpy() - y0
    N, T = len(ukey), len(years)

    V = np.full((N, T, 3, 6), np.nan, np.float32)
    QA = np.zeros((N, T, 3), np.int64); RAD = np.zeros((N, T, 3), np.int64)
    DOY = np.zeros((N, T, 3), np.float32); SEN = np.full((N, T, 3), -1, np.int8)
    VALID = np.zeros((N, T, 3), bool); TIER2 = np.zeros((N, T, 3), bool)
    ACQ = np.full((N, T, 3), -1, np.int32)
    scene_codes, scene_names = pd.factorize(pd.concat([df[f"scene_id_{s}"] for s in (1, 2, 3)]))
    n = len(df)
    for s in (1, 2, 3):
        i = s - 1
        for j, b in enumerate(BASE_BANDS):
            V[pix, t, i, j] = df[f"{b}_{s}"].to_numpy(np.float32, na_value=np.nan)
        QA[pix, t, i] = np.nan_to_num(df[f"QA_PIXEL_{s}"].to_numpy(float)).astype(np.int64)
        RAD[pix, t, i] = np.nan_to_num(df[f"QA_RADSAT_{s}"].to_numpy(float)).astype(np.int64)
        DOY[pix, t, i] = np.nan_to_num(df[f"doy_{s}"].to_numpy(float))
        if doy_shift:
            DOY[pix, t, i] = np.where(DOY[pix, t, i] > 0, (DOY[pix, t, i] + doy_shift - 1) % 365.25 + 1, 0)
        sid = df[f"scene_id_{s}"].fillna("")
        unknown = sorted(set(sid[(sid != "") & ~sid.str[:4].isin(SENSORS)].str[:4]))
        if unknown:
            raise ValueError(f"unknown sensor(s) {unknown}: the backbone only knows {SENSORS}")
        SEN[pix, t, i] = np.select([sid.str[:4].to_numpy() == x for x in SENSORS], range(len(SENSORS)), -1)
        TIER2[pix, t, i] = sid.str.endswith("T2").to_numpy()
        VALID[pix, t, i] = df[f"qa_valid_{s}"].fillna(False).to_numpy(bool)
        ACQ[pix, t, i] = scene_codes[i * n:(i + 1) * n]
    static_cov = df["coverage"].to_numpy(np.float64)[first]
    px_m, py_m = df["pixel_x"].to_numpy(np.float64)[first], df["pixel_y"].to_numpy(np.float64)[first]
    wmy = df["weighted_mean_year"].to_numpy(np.float64)[first] if has_labels else None
    px_key, py_key = df["px_key"].to_numpy()[first], df["py_key"].to_numpy()[first]
    del df

    # ── saturation: QA_RADSAT band bit, or value at the sensor+band encoding ceiling ────
    is_oli = SEN >= SENSORS.index("LC08")
    ceil = np.full((len(SENSORS), 6), np.inf, np.float32)
    ceil_info = {}
    for k, sname in enumerate(SENSORS):
        m = VALID & (SEN == k)
        for j, b in enumerate(BASE_BANDS):
            vals = V[..., j][m]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            vmax = vals.max()
            n_at = int((vals >= vmax - 1).sum())
            if n_at >= CEIL_MIN_COUNT:
                ceil[k, j] = vmax - 1
                ceil_info[f"{sname}_{b}"] = dict(ceiling=float(vmax), n_at_ceiling=n_at)
    usable = np.zeros(V.shape, bool)
    n_sat_qa = n_sat_ceil = 0
    for j in range(6):
        sat_qa = np.where(is_oli, (RAD >> RADSAT_BIT["OLI"][j]) & 1, (RAD >> RADSAT_BIT["TM"][j]) & 1) == 1
        at_ceil = (SEN >= 0) & (V[..., j] >= ceil[np.clip(SEN, 0, None), j])
        base = VALID & ~np.isnan(V[..., j])
        n_sat_qa += int((base & sat_qa).sum()); n_sat_ceil += int((base & at_ceil & ~sat_qa).sum())
        usable[..., j] = base & ~sat_qa & ~at_ceil

    # ── anomalous acquisitions: city-wide median reflectance is a robust outlier ───────
    m = VALID & ~np.isnan(V).any(axis=3)          # all QA-valid obs, ceiling values included
    acq_obs = pd.DataFrame(V[m], columns=BASE_BANDS)
    acq_obs["acq"] = ACQ[m]
    g = acq_obs.groupby("acq")
    med = g[BASE_BANDS].median()
    med = med[g.size() >= MIN_ACQ_OBS]
    zz = (med - med.median()) / (1.4826 * (med - med.median()).abs().median())
    acq_z = zz.abs().max(axis=1)
    anom_codes = acq_z.index[acq_z > Z_ANOM].to_numpy()
    ANOM = np.isin(ACQ, anom_codes)
    anomalous_list = sorted(((str(scene_names[c]), float(acq_z[c])) for c in anom_codes),
                            key=lambda x: -x[1])

    # ── rank acquisitions, choose per band ─────────────────────────────────────────────
    cloud_med = ((QA >> 8) & 3) == 2
    rank = (ANOM * 1e6 + cloud_med * 1e4 + TIER2 * 1e2
            + np.abs(DOY - MID_SUMMER_DOY) / 1e3).astype(np.float64)
    chosen = np.full((N, T, 6), -1, np.int8)
    for j in range(6):
        r = np.where(usable[..., j], rank, np.inf)
        chosen[..., j] = np.where(np.isfinite(r.min(axis=2)), r.argmin(axis=2), -1)
    obs = chosen >= 0
    ci = np.clip(chosen, 0, 2).astype(np.int64)
    bands = np.stack([np.take_along_axis(V[..., j], ci[..., j:j + 1], axis=2)[..., 0]
                      for j in range(6)], axis=2)
    bands = np.where(obs, bands, np.nan).astype(np.float64)

    # main acquisition of the year = slot used by most bands
    cnt = np.stack([(chosen == s).sum(axis=2) for s in range(3)], axis=2)
    main = np.where(obs.any(axis=2), cnt.argmax(axis=2), -1)
    mi = np.clip(main, 0, 2)[..., None]
    has = main >= 0
    doy_main = np.where(has, np.take_along_axis(DOY, mi, axis=2)[..., 0], np.nan)
    sen_main = np.where(has, np.take_along_axis(SEN, mi, axis=2)[..., 0], -1)
    anom_main = has & np.take_along_axis(ANOM, mi, axis=2)[..., 0]
    tier2_main = has & np.take_along_axis(TIER2, mi, axis=2)[..., 0]

    # reference: what NB3 would have taken (first QA-valid slot, row dropped if Blue > 15000)
    nb3_slot = np.where(VALID.any(axis=2), VALID.argmax(axis=2), -1)
    nb3_blue = np.take_along_axis(V[..., 0], np.clip(nb3_slot, 0, 2)[..., None], axis=2)[..., 0]
    nb3_ok = (nb3_slot >= 0) & (nb3_blue <= 15_000)

    bands_observed = bands.copy() if return_internals else None   # before gap filling
    # ── gap filling (NB3 interpolation) + indices ──────────────────────────────────────
    tt = np.arange(T)
    for j in range(6):
        for i in np.nonzero(obs[..., j].any(axis=1) & ~obs[..., j].all(axis=1))[0]:
            mk = obs[i, :, j]
            bands[i, :, j] = np.interp(tt, tt[mk], bands[i, mk, j])
    n_unfillable = int(np.isnan(bands).any(axis=1).sum())
    if n_unfillable and strict:
        raise ValueError("a pixel-band without any usable observation")
    if n_unfillable:                                    # dataset's per-year median, still unobserved
        for j in range(6):
            med_y = np.nanmedian(np.where(obs[..., j], bands[..., j], np.nan), axis=0)
            bad = np.isnan(bands[..., j]).all(axis=1)
            bands[bad, :, j] = med_y[None, :]
    spec = np.concatenate([bands, np.stack(compute_indices(*(bands[..., j] for j in range(6))), axis=2)],
                          axis=2)

    meta = np.concatenate([
        obs.astype(np.float32),
        np.where(has, (doy_main - MID_SUMMER_DOY) / 60.0, 0.0)[..., None],
        np.stack([(sen_main == k) for k in range(len(SENSORS))], axis=2),
        (VALID.sum(axis=2) / 3.0)[..., None], anom_main[..., None], tier2_main[..., None]],
        axis=2).astype(np.float32)                                          # (N, T, 15)

    # ── tabular features: NB3 statistics on the v3 series + static ─────────────────────
    f = {}
    mm, sd = spec.mean(axis=1), spec.std(axis=1, ddof=1)
    for j, c in enumerate(BASE_BANDS + INDEX_COLS):
        f[f"{c}_mean"], f[f"{c}_std"] = mm[:, j], sd[:, j]
    early, late = years <= EARLY_CUTOFF, years >= LATE_START
    for tag, sel in (("early", early), ("late", late)):
        for j, b in enumerate(BASE_BANDS):
            f[f"{b}_{tag}_mean"] = bands[:, sel, j].mean(1)
            f[f"{b}_{tag}_std"] = bands[:, sel, j].std(1, ddof=1)
    dd = np.diff(bands, axis=1)
    for j, b in enumerate(BASE_BANDS):
        f[f"d_{b}_mean"], f[f"d_{b}_std"] = dd[..., j].mean(1), dd[..., j].std(1, ddof=1)
    anyobs = obs.any(axis=2)
    f["has_early_data"] = anyobs[:, early].any(1).astype(float)
    f["has_late_data"] = anyobs[:, late].any(1).astype(float)
    if centre_xy is not None:
        dist = np.hypot(px_m - centre_xy[0], py_m - centre_xy[1])
        f["coverage"], f["dist_centre_rel"] = static_cov, dist / np.percentile(dist, 95)
    else:
        dist = None
        f["coverage"], f["dist_centre_rel"] = static_cov, np.full(N, np.nan)

    if has_labels:
        b = list(class_bounds)
        age = pd.cut(pd.Series(wmy), [-np.inf] + b + [np.inf], labels=range(1, len(b) + 2)).astype(int).to_numpy()
        w, w_info = label_weights(wmy, bounds=b)
    else:
        age, w, w_info = np.full(N, -1), np.ones(N, np.float32), None
    feats = pd.DataFrame({"px_key": px_key, "py_key": py_key, "age_class": age, "label_weight": w})
    if keep_xy:                                         # for distances only -- never a model input
        feats["pixel_x"], feats["pixel_y"] = px_m, py_m
    feats = pd.concat([feats, pd.DataFrame({c: f[c] for c in TAB_FEATURES})], axis=1)

    nb3_used = np.where(nb3_ok[..., None], nb3_slot[..., None], -1)
    both = obs & (nb3_used >= 0)
    stats = dict(
        ceilings=ceil_info, saturated_by_qa_radsat=n_sat_qa, saturated_by_ceiling_only=n_sat_ceil,
        anomalous_acquisitions=anomalous_list, n_acquisitions=int(len(med)),
        observed_pixel_year_bands_v3=int(obs.sum()), observed_pixel_year_bands_nb3=int(nb3_ok.sum()) * 6,
        choice_differs_from_nb3_pct=100 * float((both & (chosen != nb3_used)).sum() / both.sum()),
        mixed_date_years=int((obs.all(2) & (chosen != chosen[..., :1]).any(2)).sum()),
        years_using_anomalous_acq=int(anom_main.sum()), years_using_tier2=int(tier2_main.sum()),
        label_weights=w_info, city_radius_km=None if dist is None else float(np.percentile(dist, 95) / 1000),
        pixel_bands_filled_from_year_median=n_unfillable, years=[int(y) for y in years])
    if return_internals:
        return feats, spec.astype(np.float32), meta, stats, dict(
            V=V, usable=usable, anomalous=ANOM, obs=obs, bands_observed=bands_observed)
    return feats, spec.astype(np.float32), meta, stats


def hemisphere_from_scenes(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=[f"scene_id_{s}" for s in (1, 2, 3)]).to_pandas()
    sid = pd.concat([t[c] for c in t.columns]).dropna()
    rows = sid.str.split("_").str[2].str[3:6].astype(int)
    row = float(np.median(rows))
    return ("north" if row < WRS_EQUATOR_ROW else "south"), row


def neigh_std(spec, px, py):
    """Std of each band across the 3x3 neighbourhood per year (identical to extra_features_v4)."""
    N, T = spec.shape[:2]
    gx, gy = px - px.min() + 1, py - py.min() + 1
    H, W = gy.max() + 2, gx.max() + 2
    out = np.empty((N, T, 6), np.float32)
    for j in range(6):
        g = np.full((T, H, W), np.nan, np.float32)
        g[:, gy, gx] = spec[:, :, j].T
        s1 = np.zeros((T, H, W), np.float32); s2 = np.zeros_like(s1); c = np.zeros_like(s1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sh = np.roll(np.roll(g, dy, axis=1), dx, axis=2)
                ok = ~np.isnan(sh)
                s1 += np.where(ok, sh, 0); s2 += np.where(ok, sh * sh, 0); c += ok
        with np.errstate(all="ignore"):
            var = s2 / c - (s1 / c) ** 2
        sd = np.where(c >= 3, np.sqrt(np.clip(var, 0, None)), np.nan)
        out[:, :, j] = sd[:, gy, gx].T
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Backbone: TempCNN embedding model, ensemble as one batched network, AdaBN
# ══════════════════════════════════════════════════════════════════════════════
class TempCNNEmb(nn.Module):
    def __init__(self, c_in, n_static, t_len=42, filters=64, kernel=5, fc=256, emb=EMB_DIM, p=DROPOUT0):
        super().__init__()
        self.drops = nn.ModuleList([nn.Dropout(p) for _ in range(5)])
        layers, c = [], c_in
        for i in range(3):
            layers += [nn.Conv1d(c, filters, kernel, padding=kernel // 2), nn.BatchNorm1d(filters),
                       nn.ReLU(), self.drops[i]]
            c = filters
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(filters * t_len, fc), nn.BatchNorm1d(fc),
                                nn.ReLU(), self.drops[3])
        self.head = nn.Sequential(nn.Linear(fc + n_static, 128), nn.ReLU(), self.drops[4],
                                  nn.Linear(128, emb))
        self.log_inv_tau = nn.Parameter(torch.tensor(math.log(1 / TAU0)))   # learned temperature

    def set_dropout(self, p):
        for d in self.drops:
            d.p = p

    @property
    def tau(self):
        return float(torch.exp(-self.log_inv_tau.clamp(math.log(1.0), math.log(100.0))))

    def forward(self, seq, static):
        h = self.fc(self.convs(seq.transpose(1, 2)))
        return F.normalize(self.head(torch.cat([h, static], dim=1)), dim=1)


def supcon(z, y, inv_tau, w):
    sim = z @ z.T * inv_tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
    n_pos = pos.sum(1)
    loss = -(log_prob * pos).sum(1) / n_pos.clamp_min(1)
    keep = n_pos > 0
    return (loss[keep] * w[keep]).sum() / w[keep].sum()


def legacy_meta():
    """For backbone.pt files written before metadata was stored (ensemble run of 2026-11-10)."""
    return dict(channels=BASE_BANDS + INDEX_COLS + META_CHANNELS + NEIGH_CHANNELS, years=list(range(1984, 2026)),
                class_bounds=[1960, 1984, 2004], static_cols=list(STATIC_COLS), summer_doy_north=196,
                variant="best", n_spectral=11, n_meta=15, n_neigh=6)


def load_backbone(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    side = Path(path).with_name("backbone_meta.json")
    meta = b.get("meta") or (json.loads(side.read_text()) if side.exists() else legacy_meta())
    states = b["states"][meta["variant"]][:meta.get("n_members", len(b["states"][meta["variant"]]))]
    return dict(states=states, norm=b["norm"], arch=b["arch"], emb_dim=b["emb_dim"],
                n_channels=b["n_channels"], n_static=b["n_static"], meta=meta)

def reference_model(bb):
    a = bb["arch"]
    return TempCNNEmb(bb["n_channels"], bb["n_static"], t_len=a["t_len"], filters=a["filters"],
                      kernel=a["kernel"], fc=a["fc"], emb=bb["emb_dim"])


class EnsembleNet(nn.Module):
    """All M ensemble members as ONE batched network (grouped convs, batched matmuls).
    Units: conv blocks 0..nb-1, dense = nb, head = nb+1. BatchNorm always uses running stats."""

    def __init__(self, states, ref):
        super().__init__()
        self.M, self.nb = len(states), len(ref.convs) // 4
        st = lambda k: torch.stack([s[k].float() for s in states]).to(DEVICE)
        self.attr, self.units = [], []
        for i in range(self.nb):
            conv, bn = ref.convs[4 * i], ref.convs[4 * i + 1]
            self.attr.append(dict(pad=conv.padding[0], dil=conv.dilation[0], mode=conv.padding_mode, eps=bn.eps))
            self.units.append(self._reg(f"u{i}", dict(w=st(f"convs.{4 * i}.weight"), b=st(f"convs.{4 * i}.bias"),
                                                      g=st(f"convs.{4 * i + 1}.weight"), be=st(f"convs.{4 * i + 1}.bias")),
                                        dict(rm=st(f"convs.{4 * i + 1}.running_mean"), rv=st(f"convs.{4 * i + 1}.running_var"))))
        self.bn_eps_fc = ref.fc[2].eps
        self.units.append(self._reg("dense", dict(w=st("fc.1.weight"), b=st("fc.1.bias"), g=st("fc.2.weight"),
                                                  be=st("fc.2.bias")),
                                    dict(rm=st("fc.2.running_mean"), rv=st("fc.2.running_var"))))
        self.units.append(self._reg("head", dict(w0=st("head.0.weight"), b0=st("head.0.bias"), w3=st("head.3.weight"),
                                                 b3=st("head.3.bias")), {}))
        self.p_drop = ref.drops[0].p
        self.trainable = set()
        self.templates = [{k: v.clone() for k, v in s.items()} for s in states]

    def _reg(self, name, params, buffers):
        d = nn.ParameterDict({k: nn.Parameter(v) for k, v in params.items()})
        setattr(self, f"p_{name}", d)
        for k, v in buffers.items():
            self.register_buffer(f"{name}_{k}", v)
        return (name, d)

    def set_trainable(self, units):
        self.trainable = set(units)
        for i, (_, d) in enumerate(self.units):
            for p in d.values():
                p.requires_grad_(i in self.trainable)

    def _drop(self, x, u):
        return F.dropout(x, self.p_drop, training=self.training and u in self.trainable)

    def run(self, h, static, start=0, stop=None):
        stop = self.nb + 2 if stop is None else stop
        M = self.M
        for u in range(start, stop):
            if u < self.nb:
                name, d = self.units[u]; a = self.attr[u]
                if u == 0 and start == 0:
                    B = h.shape[0]; x = h.transpose(1, 2).repeat(1, M, 1)               # shared input
                else:
                    B = h.shape[1]; x = h.permute(1, 0, 2, 3).reshape(B, -1, h.shape[-1])
                w = d["w"]; cout = w.shape[1]
                pad = a["pad"]
                if a["mode"] != "zeros" and pad > 0:
                    x = F.pad(x, (pad, pad), mode=a["mode"]); pad = 0
                y = F.conv1d(x, w.reshape(M * cout, w.shape[2], w.shape[3]), d["b"].reshape(-1), padding=pad,
                             dilation=a["dil"], groups=M)
                rm, rv = getattr(self, f"{name}_rm").reshape(1, -1, 1), getattr(self, f"{name}_rv").reshape(1, -1, 1)
                y = (y - rm) / torch.sqrt(rv + a["eps"]) * d["g"].reshape(1, -1, 1) + d["be"].reshape(1, -1, 1)
                y = self._drop(F.relu(y), u)
                h = y.reshape(B, M, cout, -1).permute(1, 0, 2, 3)
            elif u == self.nb:
                d = self.units[u][1]
                x = h.reshape(M, h.shape[1], -1)
                y = torch.baddbmm(d["b"].unsqueeze(1), x, d["w"].transpose(1, 2))
                y = (y - self.dense_rm.unsqueeze(1)) / torch.sqrt(self.dense_rv.unsqueeze(1) + self.bn_eps_fc) \
                    * d["g"].unsqueeze(1) + d["be"].unsqueeze(1)
                h = self._drop(F.relu(y), u)
            else:
                d = self.units[u][1]
                x = torch.cat([h, static.unsqueeze(0).expand(M, -1, -1)], dim=2)
                y = self._drop(F.relu(torch.baddbmm(d["b0"].unsqueeze(1), x, d["w0"].transpose(1, 2))), u)
                h = F.normalize(torch.baddbmm(d["b3"].unsqueeze(1), y, d["w3"].transpose(1, 2)), dim=2)
        return h

    def to_states(self):
        out = [{k: v.clone() for k, v in t.items()} for t in self.templates]
        keymap = {}
        for i in range(self.nb):
            keymap[f"u{i}"] = dict(w=f"convs.{4 * i}.weight", b=f"convs.{4 * i}.bias", g=f"convs.{4 * i + 1}.weight",
                                   be=f"convs.{4 * i + 1}.bias", rm=f"convs.{4 * i + 1}.running_mean",
                                   rv=f"convs.{4 * i + 1}.running_var")
        keymap["dense"] = dict(w="fc.1.weight", b="fc.1.bias", g="fc.2.weight", be="fc.2.bias",
                               rm="fc.2.running_mean", rv="fc.2.running_var")
        keymap["head"] = dict(w0="head.0.weight", b0="head.0.bias", w3="head.3.weight", b3="head.3.bias")
        for name, d in self.units:
            for k, sk in keymap[name].items():
                v = d[k] if k in d else getattr(self, f"{name}_{k}")
                for m in range(self.M):
                    out[m][sk] = v[m].detach().cpu().to(out[m][sk].dtype)
        return out


def adabn_states(bb, X, S):
    """Re-estimate every BatchNorm's statistics on the dataset's unlabelled pixels (Li et al. 2016)."""
    out = []
    perm = np.random.default_rng(0).permutation(len(X))
    for st in bb["states"]:
        m = reference_model(bb).to(DEVICE); m.load_state_dict(st)
        bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm1d)]
        for x in bns:
            x.reset_running_stats(); x.momentum = None
        m.train()
        for x in m.modules():
            if isinstance(x, nn.Dropout):
                x.eval()
        with torch.no_grad():
            for i in range(0, len(perm), 4096):
                ii = perm[i:i + 4096]
                m(X[ii], S[ii])
        for x in bns:
            x.momentum = 0.1
        out.append({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    return out


@torch.no_grad()
def embed_all(net, X, S, chunk=2048):
    net.eval()
    return torch.cat([net.run(X[i:i + chunk], S[i:i + chunk]).cpu() for i in range(0, len(X), chunk)], dim=1)


class City:
    """A dataset prepared for the backbone: standardised series, aligned channels and years."""

    def __init__(self, path, bb, centre_latlon=None, centre_xy=None, class_bounds=None):
        self.path, meta, norm = str(path), bb["meta"], bb["norm"]
        hemi, row = hemisphere_from_scenes(path)
        self.report = dict(hemisphere_from_wrs=hemi, median_wrs_row=row)
        if centre_latlon is not None:
            hemi_c = "north" if centre_latlon[0] >= 0 else "south"
            if hemi_c != hemi:
                self.report["hemisphere_warning"] = f"WRS rows say {hemi}, centre latitude says {hemi_c}; using the centre"
            hemi = hemi_c
        self.hemisphere = hemi
        doy_shift = 0.0 if hemi == "north" else 365.25 / 2
        feats, spec, metaa, stats = preprocess_dataset(path, centre_xy=None, class_bounds=class_bounds,
                                                       doy_shift=doy_shift, strict=False, keep_xy=True)
        self.stats = stats
        self.px, self.py = feats["px_key"].to_numpy(np.int64), feats["py_key"].to_numpy(np.int64)
        self.x, self.y = feats["pixel_x"].to_numpy(float), feats["pixel_y"].to_numpy(float)
        self.coverage = feats["coverage"].to_numpy(float)
        self.dataset_class = feats["age_class"].to_numpy()          # -1 unless class_bounds given (evaluation)
        neigh = neigh_std(spec, self.px, self.py)
        N = len(self.px)
        # standardise with the backbone's source statistics, then place into the backbone's layout
        z_spec = (spec - norm["spec_mu"]) / norm["spec_sd"]
        z_neigh = np.nan_to_num((neigh - norm["neigh_mu"]) / norm["neigh_sd"], nan=0.0)
        own = dict(zip(BASE_BANDS + INDEX_COLS + META_CHANNELS + NEIGH_CHANNELS,
                       list(np.moveaxis(z_spec, 2, 0)) + list(np.moveaxis(metaa, 2, 0)) + list(np.moveaxis(z_neigh, 2, 0))))
        years_own = stats["years"]
        yb = meta["years"]
        ymap = {y: i for i, y in enumerate(years_own)}
        seq = np.zeros((N, len(yb), len(meta["channels"])), np.float32)
        missing_ch = [c for c in meta["channels"] if c not in own]
        for ci, c in enumerate(meta["channels"]):
            if c in own:
                for ti, y in enumerate(yb):
                    if y in ymap:
                        seq[:, ti, ci] = own[c][:, ymap[y]]
        self.seq = torch.from_numpy(seq).to(DEVICE)
        self.report.update(missing_channels=missing_ch, extra_channels=[c for c in own if c not in meta["channels"]],
                           missing_years=[y for y in yb if y not in ymap], dropped_years=[y for y in years_own if y not in yb],
                           n_pixels=N, pixel_bands_filled_from_year_median=stats["pixel_bands_filled_from_year_median"])
        self.allow_first_conv = bool(missing_ch)
        self.centre_latlon, self.centre_xy = centre_latlon, centre_xy
        self.cache = {}
        # pixel grid (30 m units = px/py keys)
        self.gx, self.gy = self.px - self.px.min(), self.py - self.py.min()
        self.G = np.full((self.gy.max() + 1, self.gx.max() + 1), -1, np.int64)
        self.G[self.gy, self.gx] = np.arange(N)

    def resolve_centre(self, labels_idx=None, labels_cls=None):
        if self.centre_xy is not None:
            return tuple(self.centre_xy), "input (projected)"
        if self.centre_latlon is not None:
            lat, lon = self.centre_latlon
            z0 = int((lon + 180) // 6) + 1
            cands = []
            for z in (z0, z0 - 1, z0 + 1):
                cx, cy = utm(lat, lon, z)
                dx = max(self.x.min() - cx, 0, cx - self.x.max()); dy = max(self.y.min() - cy, 0, cy - self.y.max())
                cands.append((math.hypot(dx, dy), z, (cx, cy)))
            dist, z, xy = min(cands)
            if dist > 0:
                raise ValueError("the centre lies outside the dataset in every nearby UTM zone")
            return xy, f"input (lat/lon, UTM zone {z})"
        if labels_idx is None or len(labels_idx) == 0:
            raise ValueError("no centre given and no labels to fall back on")
        old = labels_idx[labels_cls == labels_cls.min()]
        return (float(self.x[old].mean()), float(self.y[old].mean())), \
            f"fallback: centroid of the {len(old)} labelled pixels of the oldest class"

    def static(self, centre_xy, bb):
        d = np.hypot(self.x - centre_xy[0], self.y - centre_xy[1])
        vals = dict(coverage=self.coverage, dist_centre_rel=d / np.percentile(d, 95))
        cols = bb["meta"]["static_cols"]
        tab = np.stack([vals[c] for c in cols], axis=1)
        return torch.from_numpy(((tab - bb["norm"]["static_mu"]) / bb["norm"]["static_sd"]).astype(np.float32)).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 4. modelCreate: spatial structure, few-shot fine-tuning, metric learning, confidence rule
# ══════════════════════════════════════════════════════════════════════════════
def ring_offsets(R):
    rings = []
    for d in range(1, R + 1):
        ys, xs = np.mgrid[-d:d + 1, -d:d + 1]
        m = np.rint(np.hypot(xs, ys)) == d
        rings.append(np.stack([ys[m], xs[m]], axis=1))
    return rings


def spatial_curve(city, Z, rng):
    """Mean similarity of pixel pairs at distance d (rings, 30 m units) from the dataset's own
    embeddings. Neighbourhood R = last ring before the curve stops falling (one-sided 95%);
    weight(d) = excess similarity over that plateau, scaled to 1 at d = 1."""
    N = len(city.px)
    anchors = rng.choice(N, min(N, 10_000), replace=False)
    Zt = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    cap = max(2, min(city.G.shape) // 4)
    e, se = [], []
    ay, ax = city.gy[anchors], city.gx[anchors]
    for d, offs in enumerate(ring_offsets(cap), start=1):
        s_sum = np.zeros(len(anchors)); s_n = np.zeros(len(anchors))
        for dy, dx in offs:
            yy, xx = ay + dy, ax + dx
            ok = (yy >= 0) & (yy < city.G.shape[0]) & (xx >= 0) & (xx < city.G.shape[1])
            j = np.full(len(anchors), -1); j[ok] = city.G[yy[ok], xx[ok]]
            v = j >= 0
            s_sum[v] += np.einsum("ij,ij->i", Zt[anchors[v]], Zt[j[v]]); s_n[v] += 1
        have = s_n > 0
        vals = s_sum[have] / s_n[have]
        e.append(vals.mean()); se.append(vals.std(ddof=1) / math.sqrt(len(vals)))
        if d >= 2 and not (e[-2] - e[-1] > 1.645 * math.hypot(se[-2], se[-1])):
            break
    e, se = np.array(e), np.array(se)
    R = len(e) - 1                                        # last ring that was still falling
    plateau = e[-1]
    if R < 1 or e[0] <= plateau:
        return dict(R=0, weights=[], curve=e.tolist(), se=se.tolist(), plateau=float(plateau))
    w = np.clip((e[:R] - plateau) / (e[0] - plateau), 0, 1)
    return dict(R=int(R), weights=w.tolist(), curve=e.tolist(), se=se.tolist(), plateau=float(plateau))


class Neighbours:
    """Spatial neighbours within R of a set of pixels (ring index per offset)."""

    def __init__(self, city, pix, R):
        self.R = R
        offs = ring_offsets(R) if R > 0 else []
        self.ring = np.concatenate([np.full(len(o), d) for d, o in enumerate(offs)]) if offs else np.zeros(0, int)
        allo = np.concatenate(offs) if offs else np.zeros((0, 2), int)
        yy = city.gy[pix][:, None] + allo[None, :, 0]; xx = city.gx[pix][:, None] + allo[None, :, 1]
        ok = (yy >= 0) & (yy < city.G.shape[0]) & (xx >= 0) & (xx < city.G.shape[1])
        self.j = np.full(yy.shape, -1); self.j[ok] = city.G[yy[ok], xx[ok]]

    def votes(self, label, rel, K, w):
        """label/rel: per city pixel (-1 = unlabelled). Ring-normalised, distance-weighted votes."""
        n = self.j.shape[0]
        if self.R == 0:
            return np.zeros((n, K)), np.zeros(n, bool)
        jj = np.where(self.j >= 0, self.j, 0)
        lab = np.where(self.j >= 0, label[jj], -1)
        r = np.where(lab >= 0, rel[jj], 0.0)
        onehot = (lab[..., None] == np.arange(K)) * r[..., None]                    # (n, off, K)
        R1 = np.eye(self.R)[self.ring]                                               # (off, R)
        S = np.einsum("nok,or->nrk", onehot, R1); Nr = np.einsum("no,or->nr", r, R1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(Nr[..., None] > 0, S / Nr[..., None], 0.0)
        return np.einsum("nrk,r->nk", frac, np.asarray(w)), (Nr > 0).any(axis=1)


def decide(c1, c2, high, votes, has_lab):
    """The confidence rule. Returns label, tier (3 high, 2 moderate-high, 1 low), conflict flag."""
    n = len(c1); idx = np.arange(n)
    v1, v2 = votes[idx, c1], votes[idx, c2]
    label = c1.copy(); tier = np.where(high, 3, 1); conflict = np.zeros(n, bool)
    low = ~high
    some = low & has_lab & (v1 + v2 > 0)
    label[some & (v2 > v1)] = c2[some & (v2 > v1)]
    tier[some] = 2
    conflict[low & has_lab & ~(v1 + v2 > 0)] = True
    return label, tier, conflict


def balanced_batches(n_idx, y, per_class, K, rng):
    pools = [rng.permutation(n_idx[y[n_idx] == c]) for c in range(K)]
    ptr = [0] * K
    for _ in range(max(1, len(n_idx) // (per_class * K))):
        parts = []
        for c in range(K):
            if ptr[c] + per_class > len(pools[c]):
                pools[c], ptr[c] = rng.permutation(pools[c]), 0
            parts.append(pools[c][ptr[c]:ptr[c] + per_class]); ptr[c] += per_class
        yield rng.permutation(np.concatenate(parts))


def protos(z, y, w, K):
    """z (M, n, E) -> (M, K, E) label-weighted, normalised class means."""
    oh = F.one_hot(torch.as_tensor(y, device=z.device), K).float() * torch.as_tensor(np.asarray(w, np.float32), device=z.device)[:, None]
    P = torch.einsum("nk,mne->mke", oh, z) / oh.sum(0).clamp_min(1e-9)[None, :, None]
    return F.normalize(P, dim=2)


def ens_sims(z, P):
    return torch.einsum("mne,mke->nk", z, P) / z.shape[0]


def stage_units(stage, nb):
    return {1: [nb + 1], 2: [nb, nb + 1], 3: [nb - 1, nb, nb + 1], 4: [0, nb - 1, nb, nb + 1]}[stage]


def lr_mult(u, nb, eta):
    """head: LR; dense: LR/eta; last conv block: LR/eta^2; first conv block: LR/eta^3."""
    depth = {nb + 1: 0, nb: 1, nb - 1: 2, 0: 3}[u]
    return 1.0 if depth == 0 else 1.0 / eta ** depth


def finetune(states, ref, stage, lr, eta, H, S, y, w, K, epochs, seed, level, eval_cb=None):
    """Train the trainable top of the ensemble on cached activations H at `level`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EnsembleNet(states, ref)
    nb = net.nb
    units = stage_units(stage, nb)
    net.set_trainable(units)
    groups = [dict(params=list(net.units[u][1].values()), lr=lr * lr_mult(u, nb, eta)) for u in units]
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    y_t, w_t = torch.as_tensor(y, device=DEVICE), torch.as_tensor(w, device=DEVICE).float()
    per_class = int(min(FIXED["per_class"], np.bincount(y, minlength=K).min()))
    params = [p for g in groups for p in g["params"]]
    for ep in range(1, epochs + 1):
        net.train()
        for bi in balanced_batches(np.arange(len(y)), y, per_class, K, rng):
            bi_t = torch.as_tensor(bi, device=DEVICE)
            hb = H[bi_t] if level == 0 else H[:, bi_t]
            z = net.run(hb, S[bi_t], start=level)
            loss = sum(supcon(z[m], y_t[bi_t], 1.0 / TAU0, w_t[bi_t]) for m in range(net.M))
            opt.zero_grad(); loss.backward()
            sq = sum((p.grad.reshape(net.M, -1) ** 2).sum(1) for p in params)      # per-model clipping
            scale = (FIXED["grad_clip"] / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in params:
                p.grad.mul_(scale.view(-1, *[1] * (p.grad.dim() - 1)))
            opt.step()
        if eval_cb is not None:
            net.eval()
            with torch.no_grad():
                eval_cb(ep, net)
    net.eval()
    return net


def macro_f1(y, p, K):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=range(K)))


def boot_se(y, p, K, rng, B=200):
    n = len(y)
    return float(np.std([macro_f1(y[s], p[s], K) for s in (rng.integers(0, n, n) for _ in range(B))]))


def otsu(v):
    h, edges = np.histogram(v, bins="auto")
    c = (edges[:-1] + edges[1:]) / 2
    best, thr = -1, float(np.median(v))
    for i in range(1, len(h)):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (h[:i] * c[:i]).sum() / w0, (h[i:] * c[i:]).sum() / w1
        b = w0 * w1 * (m0 - m1) ** 2
        if b > best:
            best, thr = b, edges[i]
    return thr


class TransferModel:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def save(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(states=self.states, norm=self.norm, meta=self.backbone_meta, arch=self.arch,
                        classes=self.classes, config=self.config, calibration=self.calibration), folder / "transfer_model.pt")
        self.predictions.to_parquet(folder / "predictions.parquet", index=False)
        (folder / "transfer_report.json").write_text(json.dumps(self.report, indent=1, default=float))


def _context(city, bb, centre_xy):
    """Label-free, per dataset and centre: AdaBN states, embeddings, spatial curve. Cached."""
    key = tuple(np.round(centre_xy, 3))
    if key in city.cache:
        return city.cache[key]
    S = city.static(centre_xy, bb)
    if "adabn" not in city.cache:
        t0 = time.time()
        city.cache["adabn"] = adabn_states(bb, city.seq, S)
        log(f"  AdaBN on {len(city.px):,} unlabelled pixels: {time.time() - t0:.0f}s")
    ref = reference_model(bb)
    base_net, ada_net = EnsembleNet(bb["states"], ref), EnsembleNet(city.cache["adabn"], ref)
    Zb, Za = embed_all(base_net, city.seq, S), embed_all(ada_net, city.seq, S)
    Zcat = (Za.permute(1, 0, 2).reshape(len(city.px), -1) / math.sqrt(Za.shape[0])).numpy()
    curve = spatial_curve(city, Zcat, np.random.default_rng(0))
    log(f"  spatial structure: neighbourhood R = {curve['R']} px ({30 * curve['R']} m)")
    ctx = dict(S=S, ref=ref, Zb=Zb, Za=Za, curve=curve, ada_net=ada_net)
    city.cache[key] = ctx
    return ctx

def _bare_backbone(city, bb, sup, y, classes, centre, centre_src, t_start, verbose):
    """k <= BARE_MAX_K: the backbone as it is + prototypes of the labelled pixels."""
    K = len(classes)
    S = city.static(centre, bb)
    Z = embed_all(EnsembleNet(bb["states"], reference_model(bb)), city.seq, S)
    P = protos(Z[:, sup].to(DEVICE), y, np.ones(len(sup), np.float32), K)
    sims = torch.cat([ens_sims(Z[:, i:i + 8192].to(DEVICE), P).cpu() for i in range(0, Z.shape[1], 8192)]).numpy()
    d = 1 - sims
    o = np.argsort(d, axis=1); c1, c2 = o[:, 0], o[:, 1]
    idx = np.arange(len(d))
    label = c1.copy(); label[sup] = y
    conf = np.full(len(d), "bare-backbone", dtype=object); conf[sup] = "labelled"
    pred = pd.DataFrame(dict(px_key=city.px, py_key=city.py, label=classes[label], confidence=conf,
                             c1=classes[c1], c2=classes[c2], dist_c1=d[idx, c1], dist_c2=d[idx, c2],
                             margin=d[idx, c2] - d[idx, c1]))
    report = dict(mode="bare backbone", rule=f"smallest class has {int(np.bincount(y).min())} labels <= {BARE_MAX_K}",
                  city=city.report, centre=dict(xy=centre, source=centre_src), n_labels=int(len(sup)),
                  classes=classes.tolist(), label_counts=np.bincount(y, minlength=K).tolist(),
                  seconds=time.time() - t_start)
    if verbose:
        log(f"  bare backbone ({report['rule']}): {report['seconds']:.0f}s")
    return TransferModel(states=bb["states"], norm=bb["norm"], backbone_meta=bb["meta"], arch=bb["arch"],
                         classes=classes.tolist(), config=dict(mode="bare backbone"), calibration={},
                         predictions=pred, report=report, support=sup, Z=Z)


def modelCreate(data, labels, centre_latlon=None, centre_xy=None, class_bounds=None,
                backbone=DEFAULT_BACKBONE, verbose=True):
    """data: a prepared City (see prepare) or a parquet path. labels: DataFrame px_key, py_key and
    age_class (any integer classes) or year (+ class_bounds). k <= BARE_MAX_K labels in the
    smallest class -> bare backbone; otherwise the full pipeline."""
    t_start = time.time()
    bb = data.bb if isinstance(data, City) else load_backbone(backbone)
    city = data if isinstance(data, City) else City(data, bb, centre_latlon, centre_xy)
    city.bb = bb
    # ── labels ────────────────────────────────────────────────────────────────
    key = pd.Series(np.arange(len(city.px)), index=pd.MultiIndex.from_arrays([city.px, city.py]))
    li = key.reindex(pd.MultiIndex.from_arrays([labels["px_key"].to_numpy(), labels["py_key"].to_numpy()]))
    if li.isna().any():
        raise ValueError(f"{int(li.isna().sum())} labelled pixels are not in the dataset")
    sup = li.to_numpy(np.int64)
    if "age_class" in labels:
        raw_cls = labels["age_class"].to_numpy()
    elif "year" in labels and class_bounds is not None:
        raw_cls = np.searchsorted(np.asarray(class_bounds, float), labels["year"].to_numpy(float), side="left") + 1
    else:
        raise ValueError("labels need age_class, or year together with class_bounds")
    classes = np.unique(raw_cls)
    K = len(classes)
    y = np.searchsorted(classes, raw_cls)
    w = (label_weights(labels["year"].to_numpy(float), bounds=list(class_bounds))[0]
         if "year" in labels and class_bounds is not None else np.ones(len(sup), np.float32))
    centre, centre_src = city.resolve_centre(sup, raw_cls)
    if int(np.bincount(y, minlength=K).min()) <= BARE_MAX_K:
        return _bare_backbone(city, bb, sup, y, classes, centre, centre_src, t_start, verbose)
    if verbose:
        log(f"  full pipeline (smallest class has {int(np.bincount(y, minlength=K).min())} labels > {BARE_MAX_K})")
    ctx = _context(city, bb, centre)
    S, ref, curve = ctx["S"], ctx["ref"], ctx["curve"]
    M = len(bb["states"]); nb = ctx["ada_net"].nb
    E_max = 2 if SMOKE else FIXED["max_epochs"]
    rng = np.random.default_rng(12345)
    # ── spatial-block folds on the labelled pixels ───────────────────────────
    from sklearn.model_selection import StratifiedGroupKFold
    n_min = int(np.bincount(y, minlength=K).min())
    n_splits = min(5, n_min)
    blk = max(curve["R"], 1)
    groups = (city.gx[sup] // blk) * 1_000_003 + city.gy[sup] // blk
    folds = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=0).split(np.zeros(len(y)), y, groups)) \
        if n_splits >= 2 else []
    stages = [1, 2, 3] + ([4] if city.allow_first_conv else [])
    configs = [(-1, None, None), (0, None, None)] + [(1, lr, None) for lr in LRS] + \
              [(s, lr, eta) for s in stages[1:] for lr in LRS for eta in ETAS]
    level_of = {1: nb, 2: nb, 3: nb - 1, 4: 0}
    level_of[1] = nb + 1
    sup_t = torch.as_tensor(sup, device=DEVICE)
    act_cache = {}

    def acts(level, idx_t):
        k = (level, int(idx_t.numel()), int(idx_t[0]) if idx_t.numel() else -1, int(idx_t.sum()))
        if k not in act_cache:
            with torch.no_grad():
                ctx["ada_net"].eval()
                act_cache[k] = city.seq[idx_t] if level == 0 else ctx["ada_net"].run(city.seq[idx_t], S[idx_t], 0, level)
        return act_cache[k]

    # ── CV over configurations ───────────────────────────────────────────────
    cv = {}                                    # (config, epoch) -> pooled held-out predictions
    if folds:
        for fi, (tr, te) in enumerate(folds):
            for cfg in configs:
                st, lr, eta = cfg
                if st in (-1, 0):
                    Z = ctx["Zb"] if st == -1 else ctx["Za"]
                    zs = Z[:, sup].to(DEVICE)
                    P = protos(zs[:, tr], y[tr], w[tr], K)
                    cv.setdefault((cfg, 0), np.full(len(y), -1))[te] = ens_sims(zs[:, te], P).argmax(1).cpu().numpy()
                    continue
                lev = level_of[st]
                H = acts(lev, sup_t)

                def cb(ep, net, tr=tr, te=te, cfg=cfg, H=H, lev=lev):
                    hb = (lambda ii: H[torch.as_tensor(ii, device=DEVICE)] if lev == 0 else H[:, torch.as_tensor(ii, device=DEVICE)])
                    ztr = net.run(hb(tr), S[sup_t[tr]], start=lev); zte = net.run(hb(te), S[sup_t[te]], start=lev)
                    P = protos(ztr, y[tr], w[tr], K)
                    cv.setdefault((cfg, ep), np.full(len(y), -1))[te] = ens_sims(zte, P).argmax(1).cpu().numpy()

                Htr = H[torch.as_tensor(tr, device=DEVICE)] if lev == 0 else H[:, torch.as_tensor(tr, device=DEVICE)]
                finetune(city.cache["adabn"], ref, st, lr, eta, Htr, S[sup_t[tr]],
                         y[tr], w[tr], K, E_max, seed=fi, level=lev, eval_cb=cb)
        scores = {k: macro_f1(y, p, K) for k, p in cv.items() if (p >= 0).all()}
        order = lambda k: ({-1: 0, 0: 1}.get(k[0][0], k[0][0] + 1), k[1], k[0][1] or 0, -(k[0][2] or 0))
        best_k = max(scores, key=lambda k: (scores[k], -order(k)[0]))
        se = boot_se(y, cv[best_k], K, rng)
        chosen = min((k for k in scores if scores[k] >= scores[best_k] - se), key=order)
        sel = dict(best=dict(config=best_k[0], epochs=best_k[1], f1=scores[best_k]), se=se,
                   chosen=dict(config=chosen[0], epochs=chosen[1], f1=scores[chosen]), n_folds=len(folds),
                   n_candidates=len(scores))
    else:
        chosen = ((0, None, None), 0)
        sel = dict(chosen=dict(config=chosen[0], epochs=0, f1=None), note="fewer than 2 labels in some class: no CV")
    (st, lr, eta), E = chosen
    if verbose:
        log(f"  selected: stage {st}, lr {lr}, eta {eta}, epochs {E} (CV macro-F1 {sel['chosen']['f1']})")

    # ── fit on a set of labelled pixels and embed the whole dataset ──────────
    def fit_and_embed(tr):
        if st in (-1, 0):
            Z = ctx["Zb"] if st == -1 else ctx["Za"]
            return Z, (bb["states"] if st == -1 else city.cache["adabn"])
        lev = level_of[st]
        H = acts(lev, sup_t)
        Htr = H[torch.as_tensor(tr, device=DEVICE)] if lev == 0 else H[:, torch.as_tensor(tr, device=DEVICE)]
        net = finetune(city.cache["adabn"], ref, st, lr, eta, Htr, S[sup_t[tr]], y[tr], w[tr], K, E, seed=999, level=lev)
        out = []
        with torch.no_grad():
            for i in range(0, len(city.px), 2048):
                ii = torch.arange(i, min(i + 2048, len(city.px)), device=DEVICE)
                h = city.seq[ii] if lev == 0 else ctx["ada_net"].run(city.seq[ii], S[ii], 0, lev)
                out.append(net.run(h, S[ii], start=lev).cpu())
        return torch.cat(out, dim=1), net.to_states()

    def margins(Z, tr):
        P = protos(Z[:, sup[tr]].to(DEVICE), y[tr], w[tr], K)
        sims = torch.cat([ens_sims(Z[:, i:i + 8192].to(DEVICE), P).cpu() for i in range(0, Z.shape[1], 8192)])
        d = 1 - sims.numpy()
        o = np.argsort(d, axis=1)
        c1, c2 = o[:, 0], o[:, 1]
        idx = np.arange(len(d))
        delta = d[idx, c2] - d[idx, c1]
        q = (np.argsort(np.argsort(delta)) + 0.5) / len(delta)                 # margin quantile
        return d, c1, c2, delta, q

    # ── calibration: reliability of c1 by margin, and tau, from the same folds ──
    from sklearn.isotonic import IsotonicRegression
    calib = dict(source=None)
    R, wts = curve["R"], curve["weights"]
    if folds:
        held = []                                                               # per fold: data for tau search
        for fi, (tr, te) in enumerate(folds):
            Zf, _ = fit_and_embed(tr)
            d, c1, c2, delta, q = margins(Zf, tr)
            held.append((tr, te, c1, c2, q))
        q_te = np.concatenate([h[4][sup[h[1]]] for h in held])
        ok_te = np.concatenate([h[2][sup[h[1]]] == y[h[1]] for h in held])
        iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds="clip").fit(q_te, ok_te) \
            if len(np.unique(q_te)) > 1 else None
        acc0 = float(ok_te.mean())
        rel_fn = lambda qq: (np.zeros(0) if len(qq) == 0 else iso.predict(qq) if iso is not None
                             else np.full(len(qq), acc0))
        grid = np.round(np.arange(0, 1.0001, 0.01), 2)
        f1_q, preds_q = [], []
        if R > 0:
            nbrs = [Neighbours(city, sup[te], R) for (tr, te, *_r) in held]
            for qq in grid:
                yt, yp = [], []
                for (tr, te, c1, c2, q), nb_ in zip(held, nbrs):
                    lab = np.full(len(city.px), -1); rel = np.zeros(len(city.px))
                    hi = q >= qq
                    lab[hi] = c1[hi]; rel[hi] = rel_fn(q[hi])
                    lab[sup[te]] = -1; rel[sup[te]] = 0                     # held-out labels hidden
                    lab[sup[tr]] = y[tr]; rel[sup[tr]] = 1.0                # ground truth of the fold
                    v, hl = nb_.votes(lab, rel, K, wts)
                    lbl, _, _ = decide(c1[sup[te]], c2[sup[te]], q[sup[te]] >= qq, v, hl)
                    yt.append(y[te]); yp.append(lbl)
                f1_q.append(macro_f1(np.concatenate(yt), np.concatenate(yp), K)); preds_q.append(np.concatenate(yp))
            f1_q = np.array(f1_q)
            i_best = int(np.argmax(f1_q))
            q_star = float(grid[i_best])                                        # ties -> smallest quantile
            # cautious variant (reported alongside): smallest quantile within one bootstrap SE of the best,
            # i.e. hand pixels to the neighbours only as far as the CV gain exceeds its noise
            se_q = boot_se(np.concatenate(yt), preds_q[i_best], K, rng)
            q_1se = float(grid[int(np.nonzero(f1_q >= f1_q[i_best] - se_q)[0][0])])
            calib.update(source="CV (held-out labelled pixels)", q_star=q_star, q_cautious=q_1se, se=se_q,
                         f1_by_quantile=f1_q.tolist(), f1_model_only=float(f1_q[0]), f1_at_q_star=float(f1_q.max()))
        calib["iso_x"] = iso.X_thresholds_.tolist() if iso is not None else None
        calib["iso_y"] = iso.y_thresholds_.tolist() if iso is not None else None
    else:
        rel_fn = None
    # ── final model on all labelled pixels ───────────────────────────────────
    all_tr = np.arange(len(y))
    Zf, states_final = fit_and_embed(all_tr)
    d, c1, c2, delta, q = margins(Zf, all_tr)
    if calib.get("q_star") is not None:
        tau = float(np.quantile(delta, calib["q_star"]))
    else:
        tau = float(otsu(delta)); calib["source"] = "Otsu split of the dataset's margins (no usable CV)"
    if rel_fn is None:
        rel_fn = lambda qq: np.ones(len(qq))
    calib["tau"] = tau

    def apply_rule(tau_):
        high = delta >= tau_
        lab = np.full(len(city.px), -1); rel = np.zeros(len(city.px))
        lab[high] = c1[high]; rel[high] = rel_fn(q[high])
        lab[sup] = y; rel[sup] = 1.0
        low_idx = np.nonzero(~high)[0]
        votes = np.zeros((len(city.px), K)); has_lab = np.zeros(len(city.px), bool)
        if R > 0:
            for i in range(0, len(low_idx), 2000):
                ii = low_idx[i:i + 2000]
                v, hl = Neighbours(city, ii, R).votes(lab, rel, K, wts)
                votes[ii], has_lab[ii] = v, hl
        label, tier, conflict = decide(c1, c2, high, votes, has_lab)
        label[sup] = y; tier[sup] = 4
        return label, tier, conflict, votes, has_lab

    label, tier, conflict, votes, has_lab = apply_rule(tau)
    if calib.get("q_cautious") is not None and calib["q_cautious"] != calib.get("q_star"):
        calib["tau_cautious"] = float(np.quantile(delta, calib["q_cautious"]))
        label_c = apply_rule(calib["tau_cautious"])[0]
    else:
        calib["tau_cautious"] = tau; label_c = label
    # kNN on the labelled pixels (reported)
    kk = int(math.sqrt(n_min)) | 1
    Zs = Zf[:, sup].to(DEVICE)
    knn = []
    for i in range(0, Zf.shape[1], 4096):
        sims = torch.einsum("mne,mse->ns", Zf[:, i:i + 4096].to(DEVICE), Zs) / Zf.shape[0]
        top = sims.topk(min(kk, len(sup)), dim=1).indices
        knn.append(F.one_hot(torch.as_tensor(y, device=DEVICE)[top], K).sum(1).argmax(1).cpu())
    knn = torch.cat(knn).numpy()
    tiers = np.array(["", "low", "moderate-high", "high", "labelled"])
    pred = pd.DataFrame(dict(px_key=city.px, py_key=city.py, label=classes[label], confidence=tiers[tier],
                             conflict=conflict, c1=classes[c1], c2=classes[c2], dist_c1=d[np.arange(len(d)), c1],
                             dist_c2=d[np.arange(len(d)), c2], margin=delta, margin_quantile=q,
                             vote_c1=votes[np.arange(len(d)), c1], vote_c2=votes[np.arange(len(d)), c2],
                             labelled_neighbour=has_lab, knn_label=classes[knn], prototype_label=classes[c1],
                             label_cautious=classes[label_c]))
    report = dict(mode="full pipeline", city=city.report, centre=dict(xy=centre, source=centre_src), n_labels=int(len(sup)),
                  classes=classes.tolist(), label_counts=np.bincount(y, minlength=K).tolist(),
                  spatial=dict(R_px=R, R_m=30 * R, weights=wts, curve=curve["curve"]), selection=sel,
                  calibration={k: v for k, v in calib.items() if k not in ("iso_x", "iso_y")},
                  tiers={str(t): int((tier == i).sum()) for i, t in enumerate(tiers) if t},
                  conflicts=int(conflict.sum()), knn_k=kk, seconds=time.time() - t_start)
    if verbose:
        log(f"  tau source: {calib['source']}; tiers {report['tiers']}; {report['seconds']:.0f}s")
    return TransferModel(states=states_final, norm=bb["norm"], backbone_meta=bb["meta"], arch=bb["arch"],
                         classes=classes.tolist(), config=dict(stage=st, lr=lr, eta=eta, epochs=E), calibration=calib,
                         predictions=pred, report=report, support=sup, Z=Zf)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Raw data unpacking and the entry points
# ══════════════════════════════════════════════════════════════════════════════
def unpack(data, workdir):
    """A parquet file, a folder with one parquet, or a .zip containing one -> path of the parquet."""
    p = Path(data)
    if p.suffix.lower() == ".zip":
        dest = Path(workdir) / "raw"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            for m in z.infolist():
                if ".." in Path(m.filename).parts or Path(m.filename).is_absolute():
                    raise ValueError(f"unsafe path in archive: {m.filename}")
            z.extractall(dest)
        p = dest
    if p.is_dir():
        found = sorted(glob.glob(str(p / "**" / "*.parquet"), recursive=True))
        if len(found) != 1:
            raise ValueError(f"expected exactly one parquet in {p}, found {len(found)}: {found[:5]}")
        p = Path(found[0])
    if p.suffix.lower() != ".parquet":
        raise ValueError(f"not a parquet file: {p}")
    return p


def prepare(data, backbone=DEFAULT_BACKBONE, centre_latlon=None, centre_xy=None, workdir=".", class_bounds=None):
    """Unpack + preprocess a dataset for the backbone. class_bounds only to read the dataset's own
    labels (for --n / --evaluate); the preprocessing itself never uses labels."""
    bb = load_backbone(backbone)
    path = unpack(data, workdir)
    log(f"preprocessing {path}")
    city = City(path, bb, centre_latlon=centre_latlon, centre_xy=centre_xy, class_bounds=class_bounds)
    city.parquet = path
    city.bb = bb
    log(f"  {len(city.px):,} pixels, hemisphere {city.hemisphere}, missing channels {city.report['missing_channels'] or 'none'}, "
        f"missing years {city.report['missing_years'] or 'none'}")
    return city


def read_labels(path):
    p = Path(path)
    lab = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
    need = {"px_key", "py_key"}
    if not need <= set(lab.columns) or not ({"age_class", "year"} & set(lab.columns)):
        raise ValueError("labels need columns px_key, py_key and age_class or year")
    return lab


def dataset_years(city):
    """weighted_mean_year per pixel of a prepared dataset (for label weights of drawn samples)."""
    yr = pd.read_parquet(city.parquet, columns=["px_key", "py_key", "weighted_mean_year"]) \
        .groupby(["px_key", "py_key"]).first()
    return yr.reindex(pd.MultiIndex.from_arrays([city.px, city.py]))["weighted_mean_year"].to_numpy(float)


def draw_labels(city, n, seed, draw=0, years=None):
    """n labelled pixels per class from the dataset's own labels; reproducible for (seed, n, draw).
    seed 0 reproduces the draws used throughout development."""
    y_all = city.dataset_class
    if (y_all < 0).all():
        raise ValueError("the dataset has no labels here: pass --bounds (and a weighted_mean_year column)")
    rng = np.random.default_rng(seed * 1_000_003 + 10_000 * n + draw)
    classes = np.unique(y_all[y_all > 0])
    short = [int(c) for c in classes if (y_all == c).sum() < n]
    if short:
        raise ValueError(f"class(es) {short} have fewer than n = {n} labelled pixels")
    sup = np.concatenate([rng.choice(np.nonzero(y_all == c)[0], n, replace=False) for c in classes])
    years = dataset_years(city) if years is None else years
    return pd.DataFrame(dict(px_key=city.px[sup], py_key=city.py[sup], year=years[sup])), sup


def evaluate(city, bounds, out, n_values=N_VALUES, draws=10, seed=0):
    """The task's protocol: for each n, `draws` random samples of n labelled pixels per class;
    modelCreate on each; macro-F1 on every other labelled pixel; mean +/- std over draws."""
    from sklearn.metrics import f1_score
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    years, y_all = dataset_years(city), city.dataset_class
    rows = []
    for n in n_values:
        for r in range(draws):
            f = out / f"n{n}_draw{r}.json"
            if f.exists():
                rows.append(json.loads(f.read_text())); continue
            labels, sup = draw_labels(city, n, seed, r, years)
            log(f"n = {n}, draw {r}")
            m = modelCreate(city, labels, class_bounds=bounds, verbose=False)
            q = np.ones(len(y_all), bool); q[sup] = False; q &= y_all > 0
            pred = m.predictions["label"].to_numpy()
            res = dict(n=n, draw=r, seed=seed, mode=m.report["mode"], macro_f1=float(f1_score(y_all[q], pred[q], average="macro")),
                       accuracy=float((pred[q] == y_all[q]).mean()), n_scored=int(q.sum()),
                       confidence=m.predictions["confidence"].value_counts().to_dict(), seconds=m.report["seconds"])
            f.write_text(json.dumps(res, indent=1, default=float)); rows.append(res)
            log(f"  macro-F1 {res['macro_f1']:.4f} ({res['mode']}, {res['seconds']:.0f}s)")
    df = pd.DataFrame(rows)
    tab = df.groupby("n").agg(mode=("mode", "first"), draws=("macro_f1", "size"), mean_f1=("macro_f1", "mean"),
                              std_f1=("macro_f1", "std"), mean_accuracy=("accuracy", "mean")).reset_index()
    tab.to_csv(out / "evaluation_summary.csv", index=False)
    L = ["# modelCreate evaluation — macro-F1 over random draws of n labelled pixels per class\n",
         f"Dataset: {city.parquet}; seed {seed}; scored on every labelled pixel not in the draw.\n",
         "| n per class | mode | draws | macro-F1 (mean ± std) | accuracy |", "|---:|---|---:|---|---:|"]
    L += [f"| {int(t.n)} | {t.mode} | {int(t.draws)} | {t.mean_f1:.4f} ± {0 if pd.isna(t.std_f1) else t.std_f1:.4f} | {t.mean_accuracy:.4f} |"
          for t in tab.itertuples()]
    L.append(f"\nAverage over n: {tab.mean_f1.mean():.4f}; best: {tab.mean_f1.max():.4f} (n = {int(tab.n[tab.mean_f1.idxmax()])})")
    (out / "evaluation_report.md").write_text("\n".join(L) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.errorbar(np.log2(tab.n), tab.mean_f1, yerr=tab.std_f1.fillna(0), marker="o", capsize=4)
        ax.set_xticks(np.log2(tab.n), [str(int(v)) for v in tab.n])
        ax.set_xlabel("labelled pixels per class (log2 scale)"); ax.set_ylabel("macro-F1 (mean ± std)")
        ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "f1_vs_log2_n.png", dpi=150)
    except ImportError:
        pass
    print("\n".join(L))
    return tab


def save_model(m, folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(states=m.states, norm=m.norm, meta=m.backbone_meta, arch=m.arch, classes=m.classes,
                    config=m.config, calibration=m.calibration), folder / "model.pt")
    m.predictions.to_parquet(folder / "predictions.parquet", index=False)
    m.predictions.to_csv(folder / "predictions.csv", index=False)
    (folder / "report.json").write_text(json.dumps(m.report, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description="Building-age map of a dataset from a few labelled pixels.")
    ap.add_argument("--data", required=True, help="parquet file, folder with one parquet, or .zip")
    ap.add_argument("--labels", help="csv/parquet: px_key, py_key, age_class or year")
    ap.add_argument("--n", type=int, help="instead of --labels: draw n labelled pixels per class from the dataset")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the draws")
    ap.add_argument("--evaluate", action="store_true", help=f"task evaluation over n in {N_VALUES}")
    ap.add_argument("--n-values", default=",".join(map(str, N_VALUES)))
    ap.add_argument("--draws", type=int, default=10, help="random draws per n for --evaluate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--centre", help="LAT,LON of the historic centre")
    ap.add_argument("--centre-xy", help="X,Y of the centre in the dataset's projected coordinates")
    ap.add_argument("--bounds", help="class boundary years, needed when labels give years, e.g. 1960,1984,2004")
    ap.add_argument("--backbone", default=str(DEFAULT_BACKBONE))
    a = ap.parse_args()
    parse = lambda s: None if s is None else tuple(float(v) for v in s.split(","))
    t0 = time.time()
    log(f"device {DEVICE}; backbone {a.backbone}")
    if not (a.labels or a.n or a.evaluate):
        ap.error("give --labels, or --n with --bounds, or --evaluate with --bounds")
    bounds = parse(a.bounds)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    city = prepare(a.data, a.backbone, centre_latlon=parse(a.centre), centre_xy=parse(a.centre_xy), workdir=a.out,
                   class_bounds=bounds if (a.n or a.evaluate) else None)
    if a.evaluate:
        evaluate(city, bounds, a.out, tuple(int(v) for v in a.n_values.split(",")), a.draws, a.seed)
        log(f"evaluation done in {time.time() - t0:.0f}s -> {a.out}")
        return
    labels = read_labels(a.labels) if a.labels else draw_labels(city, a.n, a.seed)[0]
    m = modelCreate(city, labels, class_bounds=bounds)
    save_model(m, a.out)
    vc = m.predictions["confidence"].value_counts().to_dict()
    log(f"done in {time.time() - t0:.0f}s -> {a.out} | mode: {m.report.get('mode', 'full pipeline')} | confidence {vc}")


if __name__ == "__main__":
    main()
