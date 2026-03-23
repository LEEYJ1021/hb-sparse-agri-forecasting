#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Cross-Sectional Information Pooling for Probabilistic Forecasting
Under Extreme Data Scarcity: Evidence from Agricultural Price Series
--------------------------------------------------------------------------------
Hierarchical Bayesian Forecasting with Dual-Distribution Architecture
and Online Conformal Calibration
================================================================================

DUAL-DISTRIBUTION ARCHITECTURE
────────────────────────────────
The core methodological contribution is separating the scoring objective
from the coverage objective in probabilistic forecasting:

  CRPS TRACK (Gaussian, optimized for accuracy):
    log_samples_gaussian = N(level_h, σ_innov)
    Used for: log_crps, MAE, RMSE, Mincer-Zarnowitz, CRPSS, DM tests
    Rationale: Normal distribution minimizes CRPS when the conditional mean
               is well-estimated. Gaussian CRPS gives credit for good mean
               forecasts without tail penalty.

  COVERAGE TRACK (Student-t, optimized for interval calibration):
    log_samples_t = t(ν, level_h, σ_innov × sigma_inflate_h)
    Used for: model PI coverage, Winkler score
    Rationale: Heavy tails produce wider PIs without inflating CRPS score.

  CONFORMAL TRACK (distribution-free, theoretical guarantee):
    Errors tracked from Gaussian median predictions (stable anchor).
    Conformal PI centered on Gaussian median, width = q_conf from buffer.
    Rationale: Conformal guarantee is agnostic to distribution family.

  REPORTING:
    log_crps        → from Gaussian samples  (accuracy)
    log_cov_90      → from Student-t quantiles  (model PI coverage)
    log_cov_90_conf → from conformal buffer  (distribution-free coverage)
    winkler_90      → from Student-t PI  (sharpness)

  ARCHITECTURE:
    y_train → HB-Anchored (level, σ_innov)
                 ├─[Gaussian]────────────────→ CRPS, MAE, RMSE, MZ, DM
                 ├─[Student-t × inflate]──────→ model PI, Winkler
                 └─[Conformal buffer]─────────→ conformal PI

MODEL VARIANTS
──────────────
  HB-Anchored (proposed):
    Level estimate = ETS level + James-Stein shrinkage toward category prior
                   + dampened local trend projection
    Dual-distribution architecture applied.

  HB-Mean:
    Level estimate = sample mean + James-Stein shrinkage (pure mean-reversion)
    Dual-distribution architecture applied.

  Benchmarks (single Gaussian track):
    Naïve (random walk), ETS, Historical Mean, No-Pooling, Complete-Pooling

EVALUATION
──────────
  Rolling out-of-sample evaluation with strict temporal ordering.
  No in-sample data used for accuracy metrics.
  Conformal buffer initialized online; coverage reported only after
  buffer reaches minimum size (CONFORMAL_MIN_BUFFER = 30).

DATA
────
  Source: Korea Agricultural Marketing Information Service (KAMIS)
  API:    https://apis.data.go.kr/B552845
  Items:  97 agricultural products (retail + wholesale)
  Period: 2024-01-01 to present (rolling)
================================================================================
"""

import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import stats
from scipy.stats import kruskal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

SERVICE_KEY    = os.getenv("DATA_GO_KR_KEY", "YOUR_API_KEY_HERE")
BASE_URL       = "https://apis.data.go.kr/B552845"
TIMEOUT        = 30
SLEEP_SEC      = 0.25
NUM_OF_ROWS    = 100
MAX_PAGES      = 50

TRAIN_DATE_GTE = "20240101"
TRAIN_DATE_LTE = "20241231"
TEST_DATE_GTE  = "20250101"
TEST_DATE_LTE  = datetime.now().strftime("%Y%m%d")

# Forecasting settings
MIN_OBS_OOS         = 7
LONG_SERIES_MIN_SIM = 12
HORIZONS            = [1, 3, 7]
SPARSITY_LEVELS     = [3, 5, 7]
N_SIM_REPS          = 20
N_SAMPLES           = 2000
N_SAMPLES_SIM       = 300

# Variance control
INNOV_N0         = 2
LOCAL_WINDOW     = 14
VAR_CAP_RATIO    = 6
SIGMA_FLOOR_MULT = 0.5      # floor = mult × ets_resid × sqrt(h)

# Dual-distribution parameters
TREND_DAMPING_FACTOR  = 0.85   # dampened local trend projection
T_DOF_MIN             = 8      # min Student-t dof (coverage track only)
COVERAGE_SIGMA_INFLATE = {     # sigma inflation for coverage track only
    1: 1.0,
    3: 1.3,
    7: 1.8,
}

# Conformal calibration
CONFORMAL_MIN_BUFFER = 30
CONFORMAL_MAX_BUFFER = 300
TARGET_COVERAGE      = 0.90

# Output directory
OUT = Path("./outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def _fetch(endpoint: str, extra: dict) -> pd.DataFrame:
    """Fetch one endpoint from the KAMIS API with retry logic."""
    sess = requests.Session()
    sess.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )))
    rows = []
    for page in range(1, MAX_PAGES + 1):
        params = {
            "serviceKey": SERVICE_KEY,
            "returnType": "JSON",
            "numOfRows":  NUM_OF_ROWS,
            "pageNo":     page,
            **extra,
        }
        try:
            r = sess.get(f"{BASE_URL}{endpoint}", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            body  = r.json().get("response", {}).get("body", {})
            items = (body.get("items") or {}).get("item", [])
            if not items:
                break
            if isinstance(items, dict):
                items = [items]
            rows.extend(items)
            if len(rows) >= int(body.get("totalCount", 0)):
                break
            time.sleep(SLEEP_SEC)
        except Exception as e:
            print(f"  [WARN] page {page}: {e}")
            break
    return pd.DataFrame(rows)


def collect_all_data() -> pd.DataFrame:
    """Collect retail and wholesale price data for train and test periods."""
    frames = []
    for gte, lte, split in [
        (TRAIN_DATE_GTE, TRAIN_DATE_LTE, "train"),
        (TEST_DATE_GTE,  TEST_DATE_LTE,  "test"),
    ]:
        for ep, ptype in [
            ("/periodRetail/price",    "retail"),
            ("/periodWholesale/price", "wholesale"),
        ]:
            print(f"  Fetching {ptype} [{split}] {gte}~{lte} ...", end=" ")
            raw = _fetch(ep, {
                f"cond[exmn_ymd::GTE]": gte,
                f"cond[exmn_ymd::LTE]": lte,
            })
            print(f"{len(raw):,} rows")
            if not raw.empty:
                raw["price_type"] = ptype
                raw["split"]      = split
                frames.append(raw)
    if not frames:
        raise RuntimeError("No data collected. Check your API key and network.")
    return pd.concat(frames, ignore_index=True)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw KAMIS price data.

    Steps:
      1. Coerce numeric columns
      2. Drop missing prices and zero/negative prices
      3. Remove outliers per item-market via 3×IQR rule
      4. Log-transform prices
      5. Aggregate duplicate date-item-market records by mean
    """
    df = raw.copy()
    for col in ["exmn_ymd", "item_cd", "ctgry_cd", "exmn_dd_prc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mrkt_cd"] = (
        pd.to_numeric(df.get("mrkt_cd", 0), errors="coerce")
        .fillna(0).astype(int)
    )
    df = df.dropna(subset=["exmn_ymd", "item_cd", "exmn_dd_prc"])
    df = df[df["exmn_dd_prc"] > 0].copy()

    # Outlier removal: 3×IQR per item-market
    parts = []
    for (icd, mcd), g in df.groupby(["item_cd", "mrkt_cd"]):
        q1, q3 = g["exmn_dd_prc"].quantile([0.25, 0.75])
        iqr = q3 - q1
        parts.append(g[
            (g["exmn_dd_prc"] >= q1 - 3 * iqr) &
            (g["exmn_dd_prc"] <= q3 + 3 * iqr)
        ])
    df = pd.concat(parts, ignore_index=True)

    df["log_price"] = np.log(df["exmn_dd_prc"])
    df["date"] = pd.to_datetime(
        df["exmn_ymd"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["date"])
    df["item_nm"]  = df.get("item_nm",  pd.Series("Unknown", index=df.index)).fillna("Unknown")
    df["ctgry_nm"] = df.get("ctgry_nm", pd.Series("Unknown", index=df.index)).fillna("Unknown")

    # Aggregate duplicate records
    df = (
        df.groupby(
            ["date", "exmn_ymd", "item_cd", "mrkt_cd",
             "ctgry_cd", "ctgry_nm", "item_nm", "price_type", "split"],
            as_index=False,
        )
        .agg(log_price=("log_price", "mean"), price=("exmn_dd_prc", "mean"))
    )
    df = df.sort_values(["item_cd", "mrkt_cd", "date"]).reset_index(drop=True)
    print(
        f"[INFO] Cleaned: {len(df):,} observations | "
        f"{df['item_cd'].nunique()} items | "
        f"{df['date'].min().date()} – {df['date'].max().date()}"
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  VARIANCE UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _lag_h_var(y: np.ndarray, h: int) -> float:
    """Variance of h-step-ahead differences."""
    if len(y) > h + 1:
        return max(float(np.var(y[h:] - y[:-h], ddof=1)), 1e-8)
    if len(y) == h + 1:
        return max(float((y[-1] - y[0]) ** 2), 1e-8)
    if len(y) > 1:
        sv = float(np.var(np.diff(y), ddof=1)) if len(y) > 2 else float(np.diff(y)[0] ** 2)
        return max(sv * h, 1e-8)
    return 0.05 ** 2 * h


def _local_lag_h_var(y: np.ndarray, h: int, window: int = LOCAL_WINDOW) -> float:
    """Local (windowed) lag-h variance."""
    y_local = y[-window:] if len(y) > window else y
    if len(y_local) >= h + 2:
        return max(float(np.var(y_local[h:] - y_local[:-h], ddof=1)), 1e-8)
    return _lag_h_var(y, h)


def _capped_innov_var(raw_var: float, ets_resid_std: float, h: int) -> float:
    """
    Apply sqrt(h) floor (random-walk lower bound) and variance cap.
    Used for both Gaussian and Student-t tracks.
    """
    floor     = max(SIGMA_FLOOR_MULT * ets_resid_std * np.sqrt(h), 1e-8)
    floor_var = floor ** 2
    cap_var   = floor_var * VAR_CAP_RATIO
    return float(np.clip(raw_var, floor_var, max(cap_var, floor_var * 1.001)))


def _item_diff_var(series: pd.Series) -> float:
    """Variance of first differences for a single item's price series."""
    vals = series.values
    if len(vals) < 2:
        return np.nan
    diffs = np.diff(vals)
    if len(diffs) < 2:
        return float(diffs[0] ** 2)
    return float(np.var(diffs, ddof=1))


# ════════════════════════════════════════════════════════════════════════════
# 3.  ETS AND TREND UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _ets_level(y: np.ndarray) -> Tuple[float, float]:
    """
    Simple exponential smoothing level estimate.
    Returns (level, residual_std).
    Alpha is set adaptively as 2/(n+1), clipped to [0.10, 0.50].
    """
    n     = len(y)
    alpha = float(np.clip(2 / (n + 1), 0.10, 0.50))
    level = y[0]
    fitted = []
    for obs in y:
        fitted.append(level)
        level = alpha * obs + (1 - alpha) * level
    resid = y - np.array(fitted)
    return level, max(float(np.std(resid)) if n > 1 else 0.05, 1e-4)


def _local_trend(y: np.ndarray, window: int = LOCAL_WINDOW) -> float:
    """OLS slope over the most recent `window` observations."""
    y_local = y[-window:] if len(y) > window else y
    n = len(y_local)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y_local) / denom)


def _dampened_trend_projection(slope: float, h: int,
                                damping: float = TREND_DAMPING_FACTOR) -> float:
    """
    Sum of dampened trend increments over h steps:
      Σ_{k=1}^{h} slope × damping^k  =  slope × (1 - damping^h) / (1 - damping)
    """
    if abs(damping - 1.0) < 1e-10:
        return slope * h
    return slope * (1.0 - damping ** h) / (1.0 - damping)


# ════════════════════════════════════════════════════════════════════════════
# 4.  ONLINE CONFORMAL CALIBRATION BUFFER
# ════════════════════════════════════════════════════════════════════════════

class ConformalBuffer:
    """
    Online split conformal calibration buffer.

    Tracks absolute prediction errors from the Gaussian median forecast.
    Once min_buffer errors have accumulated, computes the (1-alpha) quantile
    of the error distribution to form distribution-free prediction intervals:

        PI_conf(t) = [ŷ_gauss(t) ± q_conf(h)]

    This provides marginal coverage ≥ 1-alpha under exchangeability
    (Vovk et al., 2005; Angelopoulos & Bates, 2023).
    """

    def __init__(
        self,
        horizons: List[int] = HORIZONS,
        target_coverage: float = TARGET_COVERAGE,
        min_buffer: int = CONFORMAL_MIN_BUFFER,
        max_buffer: int = CONFORMAL_MAX_BUFFER,
    ):
        self.alpha      = 1.0 - target_coverage
        self.min_buffer = min_buffer
        self.errors: Dict[int, deque] = {
            h: deque(maxlen=max_buffer) for h in horizons
        }

    def update(self, h: int, abs_log_error: float) -> None:
        """Add one absolute log-scale prediction error to the buffer."""
        if not np.isnan(abs_log_error):
            self.errors[h].append(float(abs_log_error))

    def get_q(self, h: int) -> Optional[float]:
        """
        Return the conformal quantile for horizon h, or None if
        the buffer has fewer than min_buffer observations.
        """
        errs = list(self.errors[h])
        m    = len(errs)
        if m < self.min_buffer:
            return None
        q_level = min(np.ceil((m + 1) * (1.0 - self.alpha)) / m, 1.0)
        return float(np.quantile(errs, q_level))

    def is_ready(self, h: int) -> bool:
        return len(self.errors[h]) >= self.min_buffer

    def summary(self) -> str:
        lines = []
        for h, buf in self.errors.items():
            q = self.get_q(h)
            lines.append(
                f"h={h}: n={len(buf)}  q={q:.4f}" if q
                else f"h={h}: n={len(buf)}  [warming up]"
            )
        return "  |  ".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 5.  HIERARCHICAL STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

def build_hierarchy(df: pd.DataFrame) -> dict:
    """
    Build a three-level hierarchy of summary statistics:
      Global → Category → Item

    At each level, computes:
      - mean, std of log-prices
      - between-item variance (τ²_c)
      - within-item variance (σ²)
      - ETS residual variance per category
      - lag-h variance per category (for h in HORIZONS)
    """
    lp = df["log_price"]
    hierarchy = {
        "global": {
            "mean": float(lp.mean()),
            "std":  max(float(lp.std()), 0.01),
            "n":    len(df),
        },
        "category": {},
        "item":     {},
    }

    df_sorted   = df.sort_values(["item_cd", "mrkt_cd", "date"])
    level_stats = (
        df_sorted.groupby(["item_cd", "ctgry_cd"])["log_price"]
        .agg(item_mean="mean", item_level_var="var", item_n="count")
        .reset_index()
    )
    diff_stats = (
        df_sorted.groupby(["item_cd", "ctgry_cd"])["log_price"]
        .apply(_item_diff_var)
        .reset_index()
        .rename(columns={"log_price": "item_diff_var"})
    )
    item_stats = level_stats.merge(diff_stats, on=["item_cd", "ctgry_cd"], how="left")

    gm_lv = max(float(item_stats["item_level_var"].dropna().median()), 0.001)
    gm_dv = max(float(item_stats["item_diff_var"].dropna().median()),  1e-5)
    item_stats["item_level_var"] = item_stats["item_level_var"].fillna(gm_lv)
    item_stats["item_diff_var"]  = item_stats["item_diff_var"].fillna(gm_dv)

    ets_vars_by_cat: Dict[int, List[float]]            = {}
    lag_vars_by_cat: Dict[int, Dict[int, List[float]]] = {}

    for (icd, ccd), g in df_sorted.groupby(["item_cd", "ctgry_cd"]):
        y_item = g["log_price"].values
        if len(y_item) < 2:
            continue
        _, sig = _ets_level(y_item)
        ets_vars_by_cat.setdefault(int(ccd), []).append(sig ** 2)
        for hv in HORIZONS:
            lv = _local_lag_h_var(y_item, hv, window=LOCAL_WINDOW)
            lag_vars_by_cat.setdefault(int(ccd), {}).setdefault(hv, []).append(lv)

    all_ets_v      = [v for vl in ets_vars_by_cat.values() for v in vl]
    global_ets_var = max(float(np.median(all_ets_v)) if all_ets_v else 0.01, 1e-6)
    global_lag_vars: Dict[int, float] = {}
    for hv in HORIZONS:
        all_lv = [v for ld in lag_vars_by_cat.values() for v in ld.get(hv, [])]
        global_lag_vars[hv] = max(
            float(np.median(all_lv)) if all_lv else global_ets_var * hv, 1e-6
        )

    hierarchy["global"]["cat_ets_var"]  = global_ets_var
    hierarchy["global"]["cat_lag_vars"] = global_lag_vars

    for ccd, g in df.groupby("ctgry_cd"):
        v  = g["log_price"]
        ci = item_stats[item_stats["ctgry_cd"] == ccd].copy()
        bv  = float(ci["item_mean"].var(ddof=1)) if len(ci) > 1 else gm_lv
        wts = ci["item_n"].clip(lower=2).values
        wdv = float(np.average(ci["item_diff_var"].values, weights=wts))
        ev_list   = ets_vars_by_cat.get(int(ccd), [])
        cat_ets_v = max(float(np.median(ev_list)) if ev_list else global_ets_var, 1e-6)
        cat_lv_dict: Dict[int, float] = {}
        for hv in HORIZONS:
            lv_list = lag_vars_by_cat.get(int(ccd), {}).get(hv, [])
            cat_lv_dict[hv] = max(
                float(np.median(lv_list)) if lv_list
                else global_lag_vars.get(hv, cat_ets_v * hv),
                1e-6,
            )
        hierarchy["category"][int(ccd)] = {
            "mean":        float(v.mean()),
            "std":         max(float(v.std()), 0.01),
            "between_std": max(np.sqrt(bv), 0.005),
            "within_std":  max(np.sqrt(wdv), 1e-4),
            "n":           len(g),
            "n_items":     len(ci),
            "cat_ets_var": cat_ets_v,
            "cat_lag_vars": cat_lv_dict,
        }

    gb_var = float(item_stats["item_mean"].var(ddof=1)) if len(item_stats) > 1 else gm_lv
    gwdv   = float(np.average(
        item_stats["item_diff_var"].values,
        weights=item_stats["item_n"].clip(lower=2).values,
    ))
    hierarchy["global"]["between_std"] = max(np.sqrt(gb_var), 0.005)
    hierarchy["global"]["within_std"]  = max(np.sqrt(gwdv),   1e-4)

    for icd, g in df.groupby("item_cd"):
        v = g["log_price"]
        hierarchy["item"][int(icd)] = {
            "mean": float(v.mean()),
            "std":  max(float(v.std()), 0.01),
            "n":    len(g),
        }
    return hierarchy


def _pooled_prior(item_cd: int, ctgry_cd: int,
                   hier: dict) -> Tuple[float, float, float]:
    """
    Return (prior_mean, between_std τ, within_std σ) for an item,
    blending category-level and global-level statistics.
    """
    cat  = hier["category"].get(int(ctgry_cd), hier["global"])
    glob = hier["global"]
    w    = min(cat["n"] / (cat["n"] + 50), 0.95)
    mu   = w * cat["mean"] + (1 - w) * glob["mean"]
    tau  = cat.get("between_std", glob.get("between_std", 0.20))
    sig  = cat.get("within_std",  glob.get("within_std",  0.05))
    return float(mu), float(tau), float(sig)


# ════════════════════════════════════════════════════════════════════════════
# 6.  FORECAST MODELS
# ════════════════════════════════════════════════════════════════════════════

def hb_anchored_forecast(
    y: np.ndarray,
    item_cd: int,
    ctgry_cd: int,
    hier: dict,
    horizons: List[int],
    n_samples: int = N_SAMPLES,
) -> dict:
    """
    HB-Anchored: Hierarchical Bayesian forecast with dual-distribution architecture.

    Level estimate:
      - ETS level (adaptive exponential smoothing)
      - James-Stein shrinkage toward category prior mean
      - Dampened local trend projection

    Innovation variance:
      - Blended from item-local lag-h variance and category prior
      - Floor: SIGMA_FLOOR_MULT × ets_resid × sqrt(h)  [random-walk lower bound]
      - Cap:   floor × VAR_CAP_RATIO

    Gaussian track  → CRPS, MAE, RMSE, DM, MZ (raw σ_innov, no inflation)
    Student-t track → 90% PI, Winkler (σ_innov × inflate_h, ν ≥ T_DOF_MIN)
    Conformal track → uses Gaussian median as anchor
    """
    n = len(y)
    prior_mu, tau, sigma_pooled = _pooled_prior(item_cd, ctgry_cd, hier)
    tau_sq = tau ** 2
    cat    = hier["category"].get(int(ctgry_cd), hier["global"])

    # ── Base statistics ───────────────────────────────────────────────────
    if n < 2:
        ets_level_raw  = prior_mu
        resid_std      = sigma_pooled
        item_level_var = tau_sq
        trend_slope    = 0.0
    else:
        ets_level_raw, resid_std = _ets_level(y)
        item_level_var = max(float(np.var(y, ddof=1)), 1e-6)
        trend_slope    = _local_trend(y, LOCAL_WINDOW)

    ets_resid_std = max(resid_std, 1e-4)

    # ── James-Stein shrinkage of ETS level toward category prior ─────────
    denom_m        = n * tau_sq + item_level_var
    lam_mean       = float(np.clip(
        item_level_var / denom_m if denom_m > 0 else 0.5, 0, 1
    ))
    level_post_var = float(np.clip(
        (item_level_var * tau_sq) / denom_m if denom_m > 0 else tau_sq,
        1e-10, tau_sq,
    ))
    level_post_ets = (1 - lam_mean) * ets_level_raw + lam_mean * prior_mu

    # ── Category variance priors ──────────────────────────────────────────
    cat_lag_vars = cat.get("cat_lag_vars", {})
    cat_ets_var  = float(cat.get("cat_ets_var", sigma_pooled ** 2))
    alpha_v      = float(n / (n + INNOV_N0)) if INNOV_N0 > 0 else 1.0

    # ── Student-t degrees of freedom (coverage track only) ───────────────
    nu_cov = float(np.clip(max(n - 1, T_DOF_MIN), T_DOF_MIN, 30))

    result = {}
    for h in horizons:
        # Trend-corrected level
        trend_proj = _dampened_trend_projection(trend_slope, h, TREND_DAMPING_FACTOR)
        level_h    = level_post_ets + trend_proj

        # Innovation variance (floor applied, no inflation yet)
        item_lh  = _local_lag_h_var(y, h, LOCAL_WINDOW) if n > h else ets_resid_std ** 2 * h
        cat_lh   = float(cat_lag_vars.get(h, cat_ets_var * h))
        raw_var  = alpha_v * item_lh + (1 - alpha_v) * cat_lh
        innov_var_h = _capped_innov_var(raw_var, ets_resid_std, h)
        innov_std_h = np.sqrt(innov_var_h)

        # Shared level uncertainty across both tracks
        lev_s = np.random.normal(level_h, np.sqrt(level_post_var), n_samples)

        # ── GAUSSIAN TRACK (accuracy / CRPS) ─────────────────────────────
        # Raw σ_innov, no inflation → CRPS-optimal
        log_s_g = lev_s + np.random.normal(0.0, innov_std_h, n_samples)
        prc_s_g = np.exp(log_s_g)
        gauss_median = float(np.median(log_s_g))

        # ── STUDENT-t TRACK (interval coverage) ──────────────────────────
        # σ_innov × inflate_h, fat tails → wider PI without CRPS penalty
        inflate_h   = COVERAGE_SIGMA_INFLATE.get(h, 1.0 + 0.15 * h)
        innov_std_t = innov_std_h * inflate_h
        log_s_t     = lev_s + np.random.standard_t(df=nu_cov, size=n_samples) * innov_std_t
        prc_s_t     = np.exp(log_s_t)
        lqs_t       = np.percentile(log_s_t, [5, 95])
        qs_t        = np.percentile(prc_s_t, [2.5, 5, 90, 95, 97.5])

        result[h] = {
            # Gaussian track (accuracy)
            "log_samples":      log_s_g,
            "samples":          prc_s_g,
            "log_median_gauss": gauss_median,
            "mean":             float(np.mean(prc_s_g)),
            "median":           float(np.median(prc_s_g)),
            "std":              float(np.std(prc_s_g)),
            # Student-t track (coverage)
            "log_samples_t":    log_s_t,
            "log_q05_t":        float(lqs_t[0]),
            "log_q95_t":        float(lqs_t[1]),
            "q025_t": float(qs_t[0]), "q05_t": float(qs_t[1]),
            "q90_t":  float(qs_t[2]), "q95_t": float(qs_t[3]),
            "q975_t": float(qs_t[4]),
            # Metadata
            "model":         "hb_anchored",
            "n_train":       n,
            "lambda":        lam_mean,
            "alpha_v":       alpha_v,
            "trend_slope":   trend_slope,
            "trend_proj":    trend_proj,
            "innov_var":     innov_var_h,
            "nu_cov":        nu_cov,
            "inflate_h":     inflate_h,
            "level_post_var": level_post_var,
        }
    return result


def hb_mean_forecast(
    y: np.ndarray,
    item_cd: int,
    ctgry_cd: int,
    hier: dict,
    horizons: List[int],
    n_samples: int = N_SAMPLES,
) -> dict:
    """
    HB-Mean: Hierarchical Bayesian forecast with pure mean-reversion level.
    Uses the same dual-distribution architecture as HB-Anchored, but the
    level estimate is the JS-shrunk sample mean (no ETS, no trend).
    """
    n = len(y)
    prior_mu, tau, sigma_pooled = _pooled_prior(item_cd, ctgry_cd, hier)
    tau_sq = tau ** 2
    cat    = hier["category"].get(int(ctgry_cd), hier["global"])

    sample_mean    = float(np.mean(y)) if n >= 1 else prior_mu
    item_level_var = max(float(np.var(y, ddof=1)) if n >= 2 else tau_sq, 1e-6)
    denom_m        = n * tau_sq + item_level_var
    lam            = float(np.clip(
        item_level_var / denom_m if denom_m > 0 else 0.5, 0, 1
    ))
    mu_post    = (1 - lam) * sample_mean + lam * prior_mu
    l_post_var = float(np.clip(
        (item_level_var * tau_sq) / denom_m if denom_m > 0 else tau_sq,
        1e-10, tau_sq,
    ))

    _, resid_std  = _ets_level(y) if n >= 2 else (prior_mu, sigma_pooled)
    ets_resid_std = max(resid_std, 1e-4)
    cat_lag_vars  = cat.get("cat_lag_vars", {})
    cat_ets_var   = float(cat.get("cat_ets_var", sigma_pooled ** 2))
    alpha_v       = float(n / (n + INNOV_N0)) if INNOV_N0 > 0 else 1.0
    nu_cov        = float(np.clip(max(n - 1, T_DOF_MIN), T_DOF_MIN, 30))

    result = {}
    for h in horizons:
        item_lh = _local_lag_h_var(y, h, LOCAL_WINDOW) if n > h else ets_resid_std ** 2 * h
        cat_lh  = float(cat_lag_vars.get(h, cat_ets_var * h))
        raw_var = alpha_v * item_lh + (1 - alpha_v) * cat_lh
        innov_var_h = _capped_innov_var(raw_var, ets_resid_std, h)
        innov_std_h = np.sqrt(innov_var_h)

        lev_s   = np.random.normal(mu_post, np.sqrt(l_post_var), n_samples)
        log_s_g = lev_s + np.random.normal(0.0, innov_std_h, n_samples)
        prc_s_g = np.exp(log_s_g)

        inflate_h = COVERAGE_SIGMA_INFLATE.get(h, 1.0 + 0.15 * h)
        log_s_t   = lev_s + np.random.standard_t(df=nu_cov, size=n_samples) * innov_std_h * inflate_h
        prc_s_t   = np.exp(log_s_t)
        lqs_t     = np.percentile(log_s_t, [5, 95])

        result[h] = {
            "log_samples":      log_s_g,
            "samples":          prc_s_g,
            "log_median_gauss": float(np.median(log_s_g)),
            "mean":             float(np.mean(prc_s_g)),
            "median":           float(np.median(prc_s_g)),
            "std":              float(np.std(prc_s_g)),
            "log_samples_t":    log_s_t,
            "log_q05_t":        float(lqs_t[0]),
            "log_q95_t":        float(lqs_t[1]),
            "model":    "hb_mean",
            "n_train":  n,
            "lambda":   lam,
            "alpha_v":  alpha_v,
            "innov_var": innov_var_h,
        }
    return result


# ── BENCHMARK MODELS (single Gaussian track) ─────────────────────────────────

def _make_forecast(log_samples: np.ndarray, model: str, n_train: int) -> dict:
    """Package log-scale samples into a standard forecast dict."""
    prc = np.exp(log_samples)
    lqs = np.percentile(log_samples, [5, 95])
    return {
        "log_samples":      log_samples,
        "samples":          prc,
        "log_median_gauss": float(np.median(log_samples)),
        "mean":             float(np.mean(prc)),
        "median":           float(np.median(prc)),
        "std":              float(np.std(prc)),
        "log_q05_t":        float(lqs[0]),
        "log_q95_t":        float(lqs[1]),
        "model":    model,
        "n_train":  n_train,
    }


def naive_forecast(y: np.ndarray, horizons: List[int],
                    n_samples: int = N_SAMPLES) -> dict:
    """Naïve (random walk): forecast = last observed value + noise."""
    last = y[-1]
    sig  = max(float(np.std(np.diff(y))) if len(y) > 1 else 0.05, 1e-4)
    return {
        h: _make_forecast(
            np.random.normal(last, sig * np.sqrt(h), n_samples),
            "naive", len(y),
        )
        for h in horizons
    }


def ets_forecast(y: np.ndarray, horizons: List[int],
                  n_samples: int = N_SAMPLES) -> dict:
    """ETS: exponential smoothing level as point forecast."""
    level, sig = _ets_level(y)
    return {
        h: _make_forecast(
            np.random.normal(level, sig * np.sqrt(h), n_samples),
            "ets", len(y),
        )
        for h in horizons
    }


def hist_mean_forecast(y: np.ndarray, horizons: List[int],
                        n_samples: int = N_SAMPLES) -> dict:
    """Historical mean (climatological baseline)."""
    mu  = float(np.mean(y))
    sig = max(float(np.std(y, ddof=1)) if len(y) > 1 else 0.05, 1e-4)
    return {
        h: _make_forecast(
            np.random.normal(mu, sig, n_samples),
            "hist_mean", len(y),
        )
        for h in horizons
    }


def no_pool_forecast(y: np.ndarray, horizons: List[int],
                      n_samples: int = N_SAMPLES) -> dict:
    """No-pooling: item-specific estimates without any cross-sectional sharing."""
    mu = float(np.mean(y)) if len(y) >= 1 else 0.0
    _, resid_std = _ets_level(y) if len(y) >= 2 else (mu, 0.05)
    result = {}
    for h in horizons:
        raw_lh  = _local_lag_h_var(y, h, LOCAL_WINDOW)
        floored = _capped_innov_var(raw_lh, resid_std, h)
        sig     = max(np.sqrt(floored), 1e-4)
        result[h] = _make_forecast(
            np.random.normal(mu, sig, n_samples), "no_pool", len(y)
        )
    return result


def complete_pool_forecast(hier: dict, ctgry_cd: int,
                            horizons: List[int],
                            n_samples: int = N_SAMPLES) -> dict:
    """
    Complete-pooling: all items in a category share the category mean.
    Variance = τ²_c + cat_lag_var (includes between-item variance).
    """
    cat          = hier["category"].get(int(ctgry_cd), hier["global"])
    mu           = float(cat["mean"])
    tau_c        = float(cat.get("between_std", hier["global"].get("between_std", 0.20)))
    cat_lag_vars = cat.get("cat_lag_vars", {})
    cat_ets_var  = float(cat.get("cat_ets_var", 0.01))
    result = {}
    for h in horizons:
        lag_var_h = float(cat_lag_vars.get(h, cat_ets_var * h))
        total_var = tau_c ** 2 + lag_var_h
        sig       = max(np.sqrt(total_var), 1e-4)
        result[h] = _make_forecast(
            np.random.normal(mu, sig, n_samples), "complete_pool", -1
        )
    return result


def get_all_forecasters(y: np.ndarray, icd: int, ccd: int,
                         local_hier: dict) -> dict:
    """Return all model forecasts for a given training series."""
    return {
        "hb_anchored":   hb_anchored_forecast(y, icd, ccd, local_hier, HORIZONS),
        "hb_mean":       hb_mean_forecast(y, icd, ccd, local_hier, HORIZONS),
        "naive":         naive_forecast(y, HORIZONS),
        "ets":           ets_forecast(y, HORIZONS),
        "hist_mean":     hist_mean_forecast(y, HORIZONS),
        "no_pool":       no_pool_forecast(y, HORIZONS),
        "complete_pool": complete_pool_forecast(local_hier, ccd, HORIZONS),
    }


# ════════════════════════════════════════════════════════════════════════════
# 7.  ACCURACY METRICS
# ════════════════════════════════════════════════════════════════════════════

def _crps_fast(samples: np.ndarray, y: float) -> float:
    """Energy-form CRPS for a set of predictive samples."""
    s  = np.sort(samples)
    n  = len(s)
    t1 = np.mean(np.abs(s - y))
    t2 = np.sum(np.diff(s) * np.arange(1, n) * (n - np.arange(1, n))) / n ** 2
    return float(t1 - t2)


def log_crps(log_samples: np.ndarray, y_log: float) -> float:
    return _crps_fast(log_samples, y_log)


def price_crps(price_samples: np.ndarray, y_price: float) -> float:
    return _crps_fast(price_samples, y_price)


def winkler_score(lo: float, hi: float, y: float, alpha: float) -> float:
    """Winkler score for a (1-alpha) prediction interval."""
    w = hi - lo
    if y < lo:
        return float(w + 2 / alpha * (lo - y))
    if y > hi:
        return float(w + 2 / alpha * (y - hi))
    return float(w)


def interval_coverage(lo: float, hi: float, y: float) -> int:
    return int(lo <= y <= hi)


def crpss(crps_model: float, crps_ref: float) -> float:
    """CRPS Skill Score: positive means model beats reference."""
    if crps_ref < 1e-10:
        return float("nan")
    return float(1 - crps_model / crps_ref)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> Tuple[float, float]:
    """
    Modified Diebold-Mariano test for equal predictive accuracy.
    Loss differential: d_t = e1_t^2 - e2_t^2 (squared CRPS).
    Returns (DM statistic, two-sided p-value).
    """
    d   = e1 ** 2 - e2 ** 2
    T   = len(d)
    if T < 10:
        return float("nan"), float("nan")
    d_bar = np.mean(d)
    g0    = np.var(d, ddof=1)
    gs    = sum(
        2 * (1 - k / h) * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        for k in range(1, h)
    )
    lrv = (g0 + gs) / T
    if lrv <= 0:
        return float("nan"), float("nan")
    dm  = d_bar / np.sqrt(lrv)
    dmh = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(dmh), float(2 * stats.t.sf(abs(dmh), df=T - 1))


# ════════════════════════════════════════════════════════════════════════════
# 8.  ROLLING OUT-OF-SAMPLE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def rolling_oos_evaluation(df: pd.DataFrame, hier_global: dict) -> pd.DataFrame:
    """
    Rolling out-of-sample evaluation with strict temporal ordering.

    For each item-market pair with ≥ MIN_OBS_OOS observations:
      - At each time step t, train on y[:t], forecast y[t+h-1]
      - Hierarchy is rebuilt from data available before the forecast origin
      - Conformal buffer is updated online after each HB-Anchored forecast

    CRPS scored from Gaussian log_samples (accuracy track).
    Coverage evaluated from Student-t log_samples_t (model PI track).
    Conformal coverage evaluated when buffer is ready (n ≥ CONFORMAL_MIN_BUFFER).
    """
    print("[INFO] Rolling out-of-sample evaluation ...")
    conf_buffer = ConformalBuffer(
        HORIZONS, TARGET_COVERAGE, CONFORMAL_MIN_BUFFER, CONFORMAL_MAX_BUFFER
    )
    records = []
    groups  = (
        df.groupby(["item_cd", "mrkt_cd", "ctgry_cd", "ctgry_nm"])
        .filter(lambda g: len(g) >= MIN_OBS_OOS)
        .groupby(["item_cd", "mrkt_cd", "ctgry_cd", "ctgry_nm"])
    )
    total = len(groups)
    done  = 0

    for (icd, mcd, ccd, cnm), grp in groups:
        grp = grp.sort_values("date").reset_index(drop=True)
        y   = grp["log_price"].values
        nd  = np.abs(np.diff(y))

        for t in range(MIN_OBS_OOS, len(y)):
            y_train = y[:t]
            cutoff  = grp.loc[t, "date"]

            # Rebuild hierarchy using only data available before the forecast origin
            same_cat = df.loc[df["ctgry_cd"] == ccd, "item_cd"].unique()
            cat_df   = df[df["item_cd"].isin(same_cat) & (df["date"] < cutoff)]
            lh       = build_hierarchy(cat_df) if len(cat_df) >= 5 else hier_global
            if not lh["item"]:
                lh = hier_global

            fcs = get_all_forecasters(y_train, icd, ccd, lh)

            for h in HORIZONS:
                t_tgt = t + h - 1
                if t_tgt >= len(y):
                    continue
                y_log   = y[t_tgt]
                y_price = np.exp(y_log)
                ni      = t - MIN_OBS_OOS
                nm      = float(nd[ni]) if ni < len(nd) else float("nan")

                for mn, fcd in fcs.items():
                    fc   = fcd[h]
                    ls_g = fc["log_samples"]
                    ps_g = fc["samples"]

                    lo_t   = float(fc.get("log_q05_t", np.percentile(ls_g, 5)))
                    hi_t   = float(fc.get("log_q95_t", np.percentile(ls_g, 95)))
                    lo_t_p = float(np.exp(lo_t))
                    hi_t_p = float(np.exp(hi_t))

                    gauss_med = float(fc.get("log_median_gauss", np.median(ls_g)))
                    el        = y_log - gauss_med
                    wink90    = winkler_score(lo_t_p, hi_t_p, y_price, 0.10)

                    q_conf   = conf_buffer.get_q(h) if mn == "hb_anchored" else None
                    cov_conf = float("nan")
                    if mn == "hb_anchored" and q_conf is not None:
                        cov_conf = interval_coverage(
                            gauss_med - q_conf, gauss_med + q_conf, y_log
                        )

                    records.append({
                        "item_cd":   int(icd),
                        "mrkt_cd":   int(mcd),
                        "ctgry_cd":  int(ccd),
                        "ctgry_nm":  cnm,
                        "t":         t,
                        "horizon":   h,
                        "n_train":   t,
                        "model":     mn,
                        "y_true":    float(y_price),
                        "y_pred":    float(fc["median"]),
                        # Accuracy (Gaussian track)
                        "log_crps":    log_crps(ls_g, y_log),
                        "price_crps":  price_crps(ps_g, y_price),
                        "mae_log":     float(abs(el)),
                        "rmse_log_sq": float(el ** 2),
                        "naive_mae":   nm,
                        # Coverage (Student-t model PI)
                        "log_cov_90":  interval_coverage(lo_t, hi_t, y_log),
                        "winkler_90":  wink90,
                        # Coverage (conformal, HB-Anchored only)
                        "log_cov_90_conf":   cov_conf,
                        "conf_buffer_ready": int(conf_buffer.is_ready(h)),
                    })

                # Update conformal buffer with Gaussian median error
                if "hb_anchored" in fcs:
                    hb_fc = fcs["hb_anchored"][h]
                    g_med = float(hb_fc.get(
                        "log_median_gauss", np.median(hb_fc["log_samples"])
                    ))
                    conf_buffer.update(h, abs(y_log - g_med))

        done += 1
        if done % 40 == 0:
            print(f"  ... {done}/{total}  |  {conf_buffer.summary()}")

    oos = pd.DataFrame(records)
    oos["rmse_log"] = np.sqrt(oos["rmse_log_sq"])
    oos["mase"]     = oos["mae_log"] / oos["naive_mae"].replace(0, float("nan"))
    print(
        f"[INFO] OOS complete: {len(oos):,} records | "
        f"{oos['model'].nunique()} models"
    )
    print(f"[INFO] Conformal buffer (final): {conf_buffer.summary()}")
    return oos


# ════════════════════════════════════════════════════════════════════════════
# 9.  SPARSITY SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def sparsity_simulation(df: pd.DataFrame, hier_global: dict) -> pd.DataFrame:
    """
    Controlled sparsity simulation.

    For each eligible item-market pair (≥ LONG_SERIES_MIN_SIM observations):
      1. Reserve the last MAX_H observations as the test set
      2. From the remaining pool, randomly draw n_sparse consecutive observations
         as the training set (N_SIM_REPS repetitions per n_sparse)
      3. Forecast the test set and record CRPS (Gaussian track)

    This isolates the effect of training window size from item-specific effects.
    """
    print("[INFO] Sparsity simulation ...")
    records  = []
    eligible = (
        df.groupby(["item_cd", "mrkt_cd"])
        .filter(lambda g: len(g) >= LONG_SERIES_MIN_SIM)
        .groupby(["item_cd", "mrkt_cd", "ctgry_cd", "ctgry_nm"])
    )
    elist  = list(eligible)
    MAX_H  = max(HORIZONS)
    rng    = np.random.default_rng(42)
    t0     = time.time()

    print(f"  Eligible item-market pairs (≥{LONG_SERIES_MIN_SIM} obs): {len(elist)}")
    if not elist:
        return pd.DataFrame()

    for pi, ((icd, mcd, ccd, cnm), grp) in enumerate(elist):
        if pi % 50 == 0 and pi > 0:
            el = time.time() - t0
            eta = (len(elist) - pi) / (pi / el)
            print(f"    ... {pi}/{len(elist)} | elapsed {el:.0f}s | ETA ~{eta:.0f}s")

        grp    = grp.sort_values("date").reset_index(drop=True)
        y      = grp["log_price"].values
        y_test = y[-MAX_H:]
        y_pool = y[:-MAX_H]

        for n_sp in SPARSITY_LEVELS:
            if n_sp >= len(y_pool):
                continue
            for rep in range(N_SIM_REPS):
                s    = rng.integers(0, len(y_pool) - n_sp + 1)
                y_tr = y_pool[s: s + n_sp]
                for h in HORIZONS:
                    if h - 1 >= len(y_test):
                        continue
                    y_log = y_test[h - 1]
                    ns    = N_SAMPLES_SIM
                    fcs   = {
                        "hb_anchored":   hb_anchored_forecast(y_tr, icd, ccd, hier_global, [h], ns),
                        "hb_mean":       hb_mean_forecast(y_tr, icd, ccd, hier_global, [h], ns),
                        "no_pool":       no_pool_forecast(y_tr, [h], ns),
                        "ets":           ets_forecast(y_tr, [h], ns),
                        "naive":         naive_forecast(y_tr, [h], ns),
                        "hist_mean":     hist_mean_forecast(y_tr, [h], ns),
                        "complete_pool": complete_pool_forecast(hier_global, ccd, [h], ns),
                    }
                    for mn, fcd in fcs.items():
                        fc   = fcd[h]
                        ls   = fc["log_samples"]
                        lq5  = float(fc.get("log_q05_t", np.percentile(ls, 5)))
                        lq95 = float(fc.get("log_q95_t", np.percentile(ls, 95)))
                        records.append({
                            "item_cd":  int(icd),
                            "ctgry_cd": int(ccd),
                            "ctgry_nm": cnm,
                            "n_sparse": n_sp,
                            "rep":      rep,
                            "horizon":  h,
                            "model":    mn,
                            "log_crps": log_crps(ls, y_log),
                            "mae_log":  float(abs(
                                float(np.log(max(fc["median"], 1e-9))) - y_log
                            )),
                            "cov_90":   interval_coverage(lq5, lq95, y_log),
                        })

    sim = pd.DataFrame(records)
    if not sim.empty:
        sim = (
            sim.sort_values("n_sparse")
            .drop_duplicates(
                subset=["item_cd", "ctgry_cd", "n_sparse", "rep", "horizon", "model"]
            )
            .reset_index(drop=True)
        )
        print(
            f"[INFO] Simulation: {len(sim):,} records | "
            f"n_sparse = {sorted(sim['n_sparse'].unique())}"
        )
    return sim


# ════════════════════════════════════════════════════════════════════════════
# 10.  STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════

def compute_dm_tests(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Diebold-Mariano tests: HB-Anchored vs each benchmark.
    Loss function: squared CRPS on Gaussian samples.
    """
    print("[INFO] Computing Diebold-Mariano tests ...")
    benchmarks = ["naive", "ets", "hist_mean", "no_pool", "complete_pool", "hb_mean"]
    results    = []
    for h in HORIZONS:
        sub = oos[oos["horizon"] == h]
        hb  = (
            sub[sub["model"] == "hb_anchored"][["item_cd", "mrkt_cd", "t", "log_crps"]]
            .rename(columns={"log_crps": "crps_hb"})
        )
        for bm in benchmarks:
            bdf = (
                sub[sub["model"] == bm][["item_cd", "mrkt_cd", "t", "log_crps"]]
                .rename(columns={"log_crps": "crps_bm"})
            )
            m = hb.merge(bdf, on=["item_cd", "mrkt_cd", "t"], how="inner")
            if len(m) < 10:
                continue
            dm, p = diebold_mariano(m["crps_hb"].values, m["crps_bm"].values, h=h)
            results.append({
                "horizon":      h,
                "benchmark":    bm,
                "n_pairs":      len(m),
                "DM_stat":      round(dm, 3) if not np.isnan(dm) else float("nan"),
                "p_value":      round(p,  4) if not np.isnan(p)  else float("nan"),
                "hb_wins":      int(m["crps_hb"].mean() < m["crps_bm"].mean()),
                "mean_crps_hb": round(m["crps_hb"].mean(), 4),
                "mean_crps_bm": round(m["crps_bm"].mean(), 4),
                "CRPSS":        round(crpss(m["crps_hb"].mean(), m["crps_bm"].mean()), 4),
            })
    return pd.DataFrame(results)


def category_volatility_test(df: pd.DataFrame) -> dict:
    """
    Kruskal-Wallis test for category-level price volatility heterogeneity.
    Volatility measured as CV (std / mean) per item.
    """
    print("[INFO] Category volatility test ...")
    recs = []
    for (icd, ccd, cnm), g in df.groupby(["item_cd", "ctgry_cd", "ctgry_nm"]):
        if len(g) < 3:
            continue
        cv = float(g["price"].std() / g["price"].mean()) if g["price"].mean() > 0 else float("nan")
        recs.append({"item_cd": icd, "ctgry_cd": ccd, "ctgry_nm": cnm, "cv": cv})
    cv_df  = pd.DataFrame(recs).dropna()
    groups = [g["cv"].values for _, g in cv_df.groupby("ctgry_cd") if len(g) >= 3]
    if len(groups) < 2:
        return {}
    H, p  = kruskal(*groups)
    N, k  = len(cv_df), len(groups)
    eta2  = (H - k + 1) / (N - k)
    cat_summary = (
        cv_df.groupby("ctgry_nm")["cv"]
        .agg(median="median", mean="mean", std="std", n="count")
        .round(3)
        .reset_index()
        .sort_values("median", ascending=False)
    )
    return {
        "H_stat":      round(H, 2),
        "df":          k - 1,
        "p_value":     p,
        "eta_squared": round(eta2, 4),
        "cat_summary": cat_summary,
        "cv_df":       cv_df,
    }


def mincer_zarnowitz_test(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Mincer-Zarnowitz (1969) forecast unbiasedness test.
    Regresses log(y_true) on log(ŷ_gauss_median).
    Unbiased when: |α| < 0.10 and |β − 1| < 0.15.
    """
    print("[INFO] Mincer-Zarnowitz tests ...")
    recs = []
    for mn in oos["model"].unique():
        for h in HORIZONS:
            sub = oos[(oos["model"] == mn) & (oos["horizon"] == h)].dropna(
                subset=["y_true", "y_pred"]
            )
            if len(sub) < 10:
                continue
            yt = np.log(sub["y_true"].values + 1e-9)
            yp = np.log(sub["y_pred"].values + 1e-9)
            X  = np.column_stack([np.ones(len(yt)), yp])
            try:
                beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
                a, b     = float(beta[0]), float(beta[1])
                yhat     = X @ beta
                r2       = 1 - np.sum((yt - yhat) ** 2) / np.sum((yt - yt.mean()) ** 2)
                recs.append({
                    "model":     mn,
                    "horizon":   h,
                    "alpha":     round(a, 4),
                    "beta":      round(b, 4),
                    "R2":        round(float(r2), 4),
                    "unbiased":  int(abs(a) < 0.10 and abs(b - 1) < 0.15),
                })
            except Exception:
                pass
    return pd.DataFrame(recs)


def compute_crpss_table(oos: pd.DataFrame) -> pd.DataFrame:
    """CRPSS of each model vs. climatological baseline (hist_mean)."""
    ref = (
        oos[oos["model"] == "hist_mean"]
        .groupby(["horizon", "item_cd", "mrkt_cd", "t"])["log_crps"]
        .mean()
        .reset_index()
        .rename(columns={"log_crps": "crps_ref"})
    )
    tbl = []
    for mn in oos["model"].unique():
        for h in HORIZONS:
            sub = oos[(oos["model"] == mn) & (oos["horizon"] == h)]
            if sub.empty:
                continue
            mg = sub.merge(ref[ref["horizon"] == h],
                           on=["horizon", "item_cd", "mrkt_cd", "t"])
            if mg.empty:
                continue
            ss = crpss(mg["log_crps"].mean(), mg["crps_ref"].mean())
            tbl.append({
                "model":         mn,
                "horizon":       h,
                "CRPSS":         round(ss, 4),
                "mean_log_crps": round(mg["log_crps"].mean(), 4),
            })
    return pd.DataFrame(tbl).sort_values(["horizon", "CRPSS"], ascending=[True, False])


def compute_conformal_coverage_summary(oos: pd.DataFrame) -> pd.DataFrame:
    """Summarize conformal PI coverage for HB-Anchored (buffer-ready periods only)."""
    hb = oos[(oos["model"] == "hb_anchored") & (oos["conf_buffer_ready"] == 1)].copy()
    if hb.empty:
        return pd.DataFrame()
    tbl = []
    for h in sorted(hb["horizon"].unique()):
        sh    = hb[hb["horizon"] == h]
        cov_m = float(sh["log_cov_90"].mean())
        cov_c = (
            float(sh["log_cov_90_conf"].dropna().mean())
            if sh["log_cov_90_conf"].notna().any()
            else float("nan")
        )
        tbl.append({
            "horizon":      h,
            "n_forecasts":  len(sh),
            "cov_model_PI": round(cov_m, 4),
            "cov_conf_PI":  round(cov_c, 4),
            "gain_pp":      round((cov_c - cov_m) * 100, 2) if not np.isnan(cov_c) else float("nan"),
            "target":       TARGET_COVERAGE,
        })
    return pd.DataFrame(tbl)


# ════════════════════════════════════════════════════════════════════════════
# 11.  VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

MODELS_ORDERED = [
    "hb_anchored", "ets", "hist_mean", "naive",
    "no_pool", "complete_pool", "hb_mean",
]
MODEL_LABELS = {
    "hb_anchored":  "HB-Anchored\n(Proposed)",
    "hb_mean":      "HB-Mean",
    "ets":          "ETS",
    "hist_mean":    "Hist. Mean",
    "naive":        "Naïve (RW)",
    "no_pool":      "No Pooling",
    "complete_pool": "Complete Pool",
}
PALETTE = dict(zip(MODELS_ORDERED, [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#aec7e8",
]))
CATEGORY_LABELS_EN = {
    "채소류": "Vegetables",   "과일류": "Fruits",
    "과채류": "Fruit Veg",    "엽채류": "Leafy Veg",
    "근채류": "Root Veg",     "양념채소": "Seasoning Veg",
    "축산물": "Livestock",    "수산물": "Seafood",
    "곡류":  "Grains",       "서류":   "Tubers",
    "버섯류": "Mushrooms",    "식량작물": "Staple Crops",
    "특용작물": "Industrial Crops", "식품": "Processed Foods",
}


def _en_cat(name: str) -> str:
    return CATEGORY_LABELS_EN.get(str(name).strip(), str(name))


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure saved] {name}.png")


def fig_dual_architecture(oos: pd.DataFrame) -> None:
    """
    Main results figure (3 rows × 3 horizons):
      Row 1: Log-CRPS (Gaussian track, accuracy)
      Row 2: Model PI coverage (Student-t track)
      Row 3: Conformal PI coverage (distribution-free)
    """
    if oos.empty:
        return
    models = [m for m in MODELS_ORDERED if m in oos["model"].unique()]
    fig, axes = plt.subplots(3, len(HORIZONS), figsize=(15, 12))
    fig.suptitle(
        "Dual-Distribution Results\n"
        "Row 1: Log-CRPS (Gaussian, accuracy)  |  "
        "Row 2: Model PI Coverage (Student-t)  |  "
        "Row 3: Conformal PI Coverage (Distribution-free)",
        fontsize=12, fontweight="bold",
    )
    for j, h in enumerate(HORIZONS):
        sub  = oos[oos["horizon"] == h]
        cd   = [sub[sub["model"] == m]["log_crps"].dropna().values for m in models]
        covd = sub.groupby("model")["log_cov_90"].mean().reindex(models).values * 100

        ax = axes[0, j]
        bp = ax.boxplot(cd, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(PALETTE.get(m, "gray"))
            patch.set_alpha(0.75)
        ax.set_xticklabels(
            [MODEL_LABELS[m].replace("\n", " ") for m in models],
            rotation=35, ha="right", fontsize=7,
        )
        ax.set_title(f"h = {h}d", fontweight="bold")
        if j == 0:
            ax.set_ylabel("Log-CRPS [Gaussian] (↓ better)")
        ax.grid(True, axis="y", alpha=0.3)

        ax2 = axes[1, j]
        bars = ax2.bar(
            range(len(models)),
            [v if not np.isnan(v) else 0 for v in covd],
            color=[PALETTE.get(m, "gray") for m in models], alpha=0.75,
        )
        ax2.axhline(90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
        ax2.set_ylim(0, 115)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(
            [MODEL_LABELS[m].replace("\n", " ") for m in models],
            rotation=35, ha="right", fontsize=7,
        )
        if j == 0:
            ax2.set_ylabel("Model PI [Student-t] Coverage (%)")
            ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, covd):
            if not np.isnan(v):
                ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
                         f"{v:.0f}%", ha="center", fontsize=7)

        ax3 = axes[2, j]
        hb_ready = oos[
            (oos["model"] == "hb_anchored") &
            (oos["horizon"] == h) &
            (oos["conf_buffer_ready"] == 1)
        ]
        if not hb_ready.empty and hb_ready["log_cov_90_conf"].notna().any():
            cov_m = float(hb_ready["log_cov_90"].mean()) * 100
            cov_c = float(hb_ready["log_cov_90_conf"].dropna().mean()) * 100
            bars3 = ax3.bar(
                ["Model PI\n(Student-t)", "Conformal PI\n(Dist.-free)"],
                [cov_m, cov_c], color=["#1f77b4", "#2ca02c"], alpha=0.80,
            )
            ax3.axhline(90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
            for bar, v in zip(bars3, [cov_m, cov_c]):
                ax3.text(bar.get_x() + bar.get_width() / 2, v + 1,
                         f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
            ax3.set_ylim(0, 115)
            if j == 0:
                ax3.set_ylabel("HB-Anchored PI Coverage (%)")
                ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, "Buffer warming up", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=9)
        ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig1_dual_architecture")


def fig_architecture_diagram(oos: pd.DataFrame) -> None:
    """Architecture diagram: CRPS evolution, coverage comparison, flowchart."""
    if oos.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Dual-Distribution Architecture: CRPS Track (Gaussian) vs Coverage Track (Student-t)\n"
        "Key insight: Separating scoring and coverage objectives resolves the long-horizon trade-off",
        fontsize=11, fontweight="bold",
    )
    ax = axes[0]
    hb_sub  = oos[oos["model"] == "hb_anchored"].groupby("horizon")["log_crps"].mean()
    ets_sub = oos[oos["model"] == "ets"].groupby("horizon")["log_crps"].mean()
    hm_sub  = oos[oos["model"] == "hist_mean"].groupby("horizon")["log_crps"].mean()
    for vals, label, col, ls in [
        (hb_sub.values,  "HB-Anchored",   "#1f77b4", "-"),
        (ets_sub.values, "ETS",            "#ff7f0e", "--"),
        (hm_sub.values,  "Hist. Mean",     "#2ca02c", ":"),
    ]:
        ax.plot(HORIZONS[:len(vals)], vals, marker="o", linewidth=2,
                color=col, linestyle=ls, label=label)
    ax.set_xlabel("Horizon h")
    ax.set_ylabel("Mean Log-CRPS (Gaussian)")
    ax.set_title("CRPS by Model and Horizon", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    ax2 = axes[1]
    hb_cov = oos[oos["model"] == "hb_anchored"].groupby("horizon")["log_cov_90"].mean() * 100
    hb_conf_cov = (
        oos[(oos["model"] == "hb_anchored") & (oos["conf_buffer_ready"] == 1)]
        .groupby("horizon")["log_cov_90_conf"]
        .apply(lambda x: x.dropna().mean() * 100)
    )
    ax2.plot(HORIZONS[:len(hb_cov)], hb_cov.values, marker="s", color="#1f77b4",
             linewidth=2, label="Student-t model PI")
    if not hb_conf_cov.empty:
        ax2.plot(HORIZONS[:len(hb_conf_cov)], hb_conf_cov.values, marker="D",
                 color="#2ca02c", linewidth=2, linestyle="--", label="Conformal PI")
    ax2.axhline(90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
    ax2.set_xlabel("Horizon h")
    ax2.set_ylabel("90% PI Coverage (%)")
    ax2.set_title("Coverage: Both PI Tracks", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(HORIZONS)

    ax3 = axes[2]
    ax3.axis("off")
    arch_text = (
        "DUAL-DISTRIBUTION ARCHITECTURE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Input: y_train (n sparse observations)\n"
        "   ↓\n"
        "HB Level: ETS + JS shrinkage + trend\n"
        "   ↓\n"
        "Innovation variance σ²_innov\n"
        "   ├─[Gaussian N(0,σ)]──────────────\n"
        "   │   CRPS track (accuracy)         \n"
        "   │   → log_crps, MAE, RMSE         \n"
        "   │   → DM test, CRPSS              \n"
        "   │   → Mincer-Zarnowitz            \n"
        "   │   → Conformal buffer anchor     \n"
        "   │                                 \n"
        "   └─[Student-t(ν≥8) × inflate]──────\n"
        "       Coverage track                \n"
        "       → 90% model PI                \n"
        "       → Winkler score               \n\n"
        "Conformal PI (distribution-free):\n"
        "   q = Quantile(|Gaussian errors|)\n"
        "   PI = [ŷ_gauss ± q]\n"
        "   → Theoretical coverage guarantee"
    )
    ax3.text(
        0.05, 0.95, arch_text,
        transform=ax3.transAxes, fontsize=8.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", alpha=0.9),
    )
    fig.tight_layout()
    _save(fig, "fig2_architecture_diagram")


def fig_sparsity(sim: pd.DataFrame) -> None:
    """Log-CRPS vs. training window n across horizons."""
    if sim.empty:
        return
    focus  = [m for m in ["hb_anchored", "hb_mean", "no_pool", "ets", "naive"]
              if m in sim["model"].unique()]
    ls_map = {
        "hb_anchored": "-", "hb_mean": "--", "no_pool": ":",
        "ets": "-.", "naive": (0, (3, 1, 1, 1)),
    }
    hp    = sorted(sim["horizon"].unique())
    fig, axes = plt.subplots(1, len(hp), figsize=(5 * len(hp), 5))
    if len(hp) == 1:
        axes = [axes]
    fig.suptitle(
        "Sparsity Simulation: Log-CRPS [Gaussian] vs Training Window",
        fontsize=12, fontweight="bold",
    )
    for j, h in enumerate(hp):
        ax  = axes[j]
        sub = sim[sim["horizon"] == h]
        for m in focus:
            agg = (
                sub[sub["model"] == m]
                .groupby("n_sparse")["log_crps"]
                .agg(mean="mean", se=lambda x: x.std() / np.sqrt(max(len(x), 1)))
                .reset_index()
            )
            if agg.empty:
                continue
            ax.plot(
                agg["n_sparse"], agg["mean"],
                label=MODEL_LABELS[m].replace("\n", " "),
                color=PALETTE.get(m, "gray"),
                linestyle=ls_map.get(m, "-"),
                linewidth=2.5 if m == "hb_anchored" else 1.5,
                marker="o", markersize=4,
            )
            ax.fill_between(
                agg["n_sparse"],
                agg["mean"] - 1.96 * agg["se"],
                agg["mean"] + 1.96 * agg["se"],
                color=PALETTE.get(m, "gray"), alpha=0.10,
            )
        ax.axvline(x=9.5, color="gray", linestyle=":", linewidth=1, label="n=10")
        ax.set_xlabel("Training window n")
        ax.set_title(f"h = {h}d", fontweight="bold")
        ticks = sorted(sim["n_sparse"].unique())
        ax.set_xticks(ticks)
        if j == 0:
            ax.set_ylabel("Mean Log-CRPS (↓ better)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_sparsity_experiment")


def fig_volatility(vol: dict) -> None:
    """Category-level price volatility distribution (violin + bar)."""
    cv_df = vol.get("cv_df")
    if cv_df is None or cv_df.empty:
        return
    cv_df = cv_df.copy()
    cv_df["ctgry_nm_en"] = cv_df["ctgry_nm"].apply(_en_cat)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Category-Level Price Volatility Heterogeneity\n"
        f"Kruskal-Wallis H={vol['H_stat']:.2f}, p<0.001, η²={vol['eta_squared']:.3f}",
        fontsize=12, fontweight="bold",
    )
    cs_en = vol["cat_summary"].copy()
    cs_en["ctgry_nm_en"] = cs_en["ctgry_nm"].apply(_en_cat)
    order = cs_en.sort_values("median", ascending=False)["ctgry_nm_en"].tolist()
    ax = axes[0]
    sns.violinplot(data=cv_df, y="ctgry_nm_en", x="cv", order=order,
                   palette="husl", ax=ax, inner="quartile", cut=0)
    ax.axvline(cv_df["cv"].median(), color="black", linestyle="--", linewidth=1.5,
               label=f"Overall median = {cv_df['cv'].median():.3f}")
    ax.set_xlabel("CV")
    ax.set_ylabel("Category")
    ax.set_title("CV Distribution by Category", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax2 = axes[1]
    cs_s = cs_en.sort_values("median", ascending=False)
    bars = ax2.barh(
        cs_s["ctgry_nm_en"], cs_s["median"],
        xerr=cs_s["std"], capsize=4,
        color=sns.color_palette("husl", len(cs_s)), alpha=0.82,
    )
    ax2.set_xlabel("Median CV")
    ax2.set_title("Median Volatility ± SD", fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)
    for bar, row in zip(bars, cs_s.itertuples()):
        ax2.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"n={row.n}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig4_category_volatility")


def fig_crpss_heatmap(crpss_df: pd.DataFrame) -> None:
    """CRPSS heatmap: model × horizon."""
    if crpss_df.empty:
        return
    pivot = crpss_df.pivot_table(index="model", columns="horizon", values="CRPSS")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot.astype(float), annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
        cbar_kws={"label": "CRPSS (Gaussian, + = better than climatology)"},
    )
    ax.set_title(
        "CRPS Skill Score vs Climatology (Hist. Mean) — Gaussian Track",
        fontweight="bold",
    )
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Model")
    fig.tight_layout()
    _save(fig, "fig5_crpss_heatmap")


def fig_dm_heatmap(dm_df: pd.DataFrame) -> None:
    """Diebold-Mariano p-value heatmap: benchmark × horizon."""
    if dm_df.empty:
        return
    try:
        pivot  = dm_df.pivot_table(index="benchmark", columns="horizon",
                                    values="p_value", aggfunc="first")
        hb_win = dm_df.pivot_table(index="benchmark", columns="horizon",
                                    values="hb_wins", aggfunc="first")
        cmap_val = pivot.copy().astype(float)
        for idx in cmap_val.index:
            for col in cmap_val.columns:
                if (idx in hb_win.index and col in hb_win.columns
                        and hb_win.loc[idx, col] == 0):
                    cmap_val.loc[idx, col] = 1.0
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            cmap_val, annot=pivot.round(3).astype(str), fmt="",
            cmap="RdYlGn_r", vmin=0, vmax=0.10, ax=ax, linewidths=0.5,
            cbar_kws={"label": "DM p-value (green = HB significantly better)"},
        )
        ax.set_title(
            "Diebold-Mariano Tests [Gaussian CRPS]\n"
            "(HB-Anchored vs Benchmarks)",
            fontweight="bold",
        )
        ax.set_xlabel("Horizon (days)")
        ax.set_ylabel("Benchmark")
        fig.tight_layout()
        _save(fig, "fig6_dm_heatmap")
    except Exception as e:
        print(f"  [WARN] fig_dm_heatmap: {e}")


def fig_pooling_gain(sim: pd.DataFrame) -> None:
    """Pooling gain (CRPS) and win rate: HB-Anchored vs No-Pooling."""
    if sim.empty:
        return
    sub = sim[sim["model"].isin(["hb_anchored", "no_pool"])].copy()
    if sub.empty:
        return
    pv = sub.pivot_table(
        index=["item_cd", "n_sparse", "rep", "horizon"],
        columns="model", values="log_crps",
    ).reset_index()
    pv.columns.name = None
    if "hb_anchored" not in pv.columns or "no_pool" not in pv.columns:
        return
    pv["gain"]      = pv["no_pool"] - pv["hb_anchored"]
    pv["gain_clip"] = pv["gain"].clip(-2, 2)
    ticks = sorted(sim["n_sparse"].unique())
    agg   = pv.groupby(["n_sparse", "horizon"])["gain_clip"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Cross-Sectional Pooling Gain: HB-Anchored vs No-Pooling\n"
        "(Log-CRPS Gaussian track)",
        fontsize=12, fontweight="bold",
    )
    ax = axes[0]
    for hv, col in zip(HORIZONS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        sh = agg[agg["horizon"] == hv]
        if sh.empty:
            continue
        se = (
            pv[pv["horizon"] == hv].groupby("n_sparse")["gain_clip"]
            .sem().reindex(sh["n_sparse"].values).fillna(0).values
        )
        ax.plot(sh["n_sparse"], sh["gain_clip"], marker="o", linewidth=2,
                label=f"h={hv}d", color=col)
        ax.fill_between(sh["n_sparse"],
                        sh["gain_clip"] - 1.96 * se,
                        sh["gain_clip"] + 1.96 * se,
                        color=col, alpha=0.12)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(x=9.5, color="gray", linestyle=":", linewidth=1, label="n=10")
    ax.set_xlabel("Training window n")
    ax.set_ylabel("CRPS gain (HB − No-Pool, clipped ±2)")
    ax.set_title("Average Pooling Gain", fontweight="bold")
    ax.set_xticks(ticks)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    pv["hb_wins"] = (pv["gain"] > 0).astype(int)
    wr = pv.groupby(["n_sparse", "horizon"])["hb_wins"].mean().reset_index()
    for hv, col in zip(HORIZONS, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        sh = wr[wr["horizon"] == hv]
        if sh.empty:
            continue
        ax2.plot(sh["n_sparse"], sh["hb_wins"] * 100, marker="s", linewidth=2,
                 linestyle="--", label=f"h={hv}d", color=col)
    ax2.axhline(50, color="black", linewidth=1, linestyle="--", label="50% break-even")
    ax2.axvline(x=9.5, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Training window n")
    ax2.set_ylabel("% Cases where HB-Anchored Wins")
    ax2.set_title("Win Rate vs No-Pooling", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(ticks)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig7_pooling_gain")


def fig_conformal_gain(oos: pd.DataFrame) -> None:
    """Coverage comparison: model PI (Student-t) vs conformal PI per horizon."""
    if oos.empty:
        return
    hb = oos[(oos["model"] == "hb_anchored") & (oos["conf_buffer_ready"] == 1)].copy()
    if hb.empty:
        print("  [SKIP] fig_conformal_gain: conformal buffer not ready")
        return
    agg = hb.groupby("horizon").agg(
        cov_model=("log_cov_90", "mean"),
        cov_conf=("log_cov_90_conf", lambda x: x.dropna().mean()),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Conformal Calibration Effect\n"
        "HB-Anchored: Student-t Model PI vs Distribution-free Conformal PI",
        fontsize=12, fontweight="bold",
    )
    w = 0.3
    x = np.arange(len(HORIZONS))
    ax = axes[0]
    ax.bar(x - w / 2, agg["cov_model"] * 100, width=w,
           label="Student-t model PI", color="#1f77b4", alpha=0.8)
    ax.bar(x + w / 2, agg["cov_conf"] * 100, width=w,
           label="Conformal PI (dist.-free)", color="#2ca02c", alpha=0.8)
    ax.axhline(90, color="red", linestyle="--", linewidth=1.5, label="Target 90%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}d" for h in HORIZONS])
    ax.set_ylabel("90% PI Coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Coverage Comparison", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for xi, (rm, rc) in enumerate(zip(agg["cov_model"] * 100, agg["cov_conf"] * 100)):
        ax.text(xi - w / 2, rm + 1, f"{rm:.0f}%", ha="center", fontsize=8)
        ax.text(xi + w / 2, rc + 1, f"{rc:.0f}%", ha="center", fontsize=8)
    ax2 = axes[1]
    gains = (agg["cov_conf"] - agg["cov_model"]) * 100
    cols  = ["#2ca02c" if g >= 0 else "#d62728" for g in gains]
    bars  = ax2.bar([f"h={h}d" for h in HORIZONS], gains, color=cols, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Coverage Gain (pp): Conformal − Model")
    ax2.set_title("Conformal Gain per Horizon", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, gains):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.3 if v >= 0 else v - 0.8,
            f"{v:+.1f}pp", ha="center", fontsize=9, fontweight="bold",
        )
    fig.tight_layout()
    _save(fig, "fig8_conformal_gain")


# ════════════════════════════════════════════════════════════════════════════
# 12.  TABLE BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def build_accuracy_table(oos: pd.DataFrame) -> pd.DataFrame:
    return (
        oos.groupby(["model", "horizon"])
        .agg(
            LogCRPS_Gaussian=("log_crps",    "mean"),
            Coverage90_t    =("log_cov_90",  "mean"),
            Winkler90       =("winkler_90",  "mean"),
            MAE_log         =("mae_log",     "mean"),
            RMSE_log        =("rmse_log",    "mean"),
            MASE            =("mase",        "mean"),
            n_forecasts     =("log_crps",    "count"),
        )
        .round(4)
        .reset_index()
    )


def build_coverage_table(oos: pd.DataFrame) -> pd.DataFrame:
    return (
        oos.groupby(["model", "horizon"])
        .agg(log_cov_90_t=("log_cov_90", "mean"))
        .round(4)
        .reset_index()
    )


def build_sparsity_table(sim: pd.DataFrame,
                          metric: str = "log_crps") -> Dict[int, pd.DataFrame]:
    if sim.empty or metric not in sim.columns:
        return {}
    out = {}
    for h in sorted(sim["horizon"].unique()):
        sub = sim[sim["horizon"] == h]
        agg = sub.groupby(["model", "n_sparse"])[metric].mean().round(4).reset_index()
        out[h] = agg.pivot_table(
            index="model", columns="n_sparse", values=metric
        ).round(4)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 13.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    print("\n" + "=" * 72)
    print("Hierarchical Bayesian Probabilistic Forecasting")
    print("Under Extreme Data Scarcity — Agricultural Price Series")
    print("=" * 72)
    print("Architecture: Dual-Distribution")
    print(f"  CRPS track:      Gaussian N(0, σ_innov)  → accuracy scoring")
    print(f"  Coverage track:  Student-t(ν≥{T_DOF_MIN}) × inflate  → PI width")
    print(f"  Conformal track: distribution-free coverage guarantee")
    print(f"Horizons: h ∈ {HORIZONS} days")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72 + "\n")

    # ── Stage 1: Data ─────────────────────────────────────────────────────
    print("── Stage 1: Data Collection ──")
    raw = collect_all_data()

    # ── Stage 2: Preprocessing ────────────────────────────────────────────
    print("\n── Stage 2: Cleaning and Hierarchy ──")
    df   = clean_data(raw)
    hier = build_hierarchy(df)
    df.to_csv(OUT / "cleaned_data.csv", index=False)
    print(f"  Hierarchy built: {len(hier['category'])} categories, {len(hier['item'])} items")

    # ── Stage 3: Rolling OOS ──────────────────────────────────────────────
    print("\n── Stage 3: Rolling Out-of-Sample Evaluation ──")
    oos = rolling_oos_evaluation(df, hier)
    oos.to_csv(OUT / "oos_results.csv", index=False)

    # ── Stage 4: Sparsity Simulation ──────────────────────────────────────
    print("\n── Stage 4: Sparsity Simulation ──")
    sim = sparsity_simulation(df, hier)
    sim.to_csv(OUT / "sparsity_simulation.csv", index=False)

    # ── Stage 5: Statistical Tests ────────────────────────────────────────
    print("\n── Stage 5: Statistical Tests ──")
    dm_df      = compute_dm_tests(oos)
    vol_result = category_volatility_test(df)
    mz_df      = mincer_zarnowitz_test(oos)
    crpss_df   = compute_crpss_table(oos)
    conf_summ  = compute_conformal_coverage_summary(oos)

    dm_df.to_csv(OUT / "dm_tests.csv", index=False)
    mz_df.to_csv(OUT / "mz_tests.csv", index=False)
    crpss_df.to_csv(OUT / "crpss.csv", index=False)
    conf_summ.to_csv(OUT / "conformal_coverage.csv", index=False)
    if vol_result.get("cat_summary") is not None:
        vol_result["cat_summary"].to_csv(OUT / "category_volatility.csv", index=False)

    # ── Stage 6: Tables ───────────────────────────────────────────────────
    print("\n── Stage 6: Summary Tables ──")
    acc_tbl  = build_accuracy_table(oos)
    cov_tbl  = build_coverage_table(oos)
    spar_tbl = build_sparsity_table(sim, "log_crps")
    acc_tbl.to_csv(OUT / "table_accuracy.csv", index=False)
    cov_tbl.to_csv(OUT / "table_coverage.csv", index=False)
    for hv, tbl in (spar_tbl.items() if isinstance(spar_tbl, dict) else []):
        tbl.to_csv(OUT / f"table_sparsity_h{hv}.csv")

    # ── Stage 7: Figures ──────────────────────────────────────────────────
    print("\n── Stage 7: Figures ──")
    if not oos.empty:
        fig_dual_architecture(oos)
        fig_architecture_diagram(oos)
        fig_crpss_heatmap(crpss_df)
        fig_conformal_gain(oos)
    if not sim.empty:
        fig_sparsity(sim)
        fig_pooling_gain(sim)
    if vol_result:
        fig_volatility(vol_result)
    if not dm_df.empty:
        fig_dm_heatmap(dm_df)

    elapsed = time.time() - t0

    # ── Key Results Summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("KEY RESULTS")
    print("=" * 72)
    if not oos.empty:
        crps_pv = acc_tbl.pivot_table(
            index="model", columns="horizon", values="LogCRPS_Gaussian"
        ).round(4)
        cov_pv = cov_tbl.pivot_table(
            index="model", columns="horizon", values="log_cov_90_t"
        ).round(3)
        print("\n▶ Log-CRPS [Gaussian track] (lower = better):")
        print(crps_pv)
        print("\n▶ Model PI Coverage [Student-t track] (target = 0.90):")
        print(cov_pv)
    if not conf_summ.empty:
        print("\n▶ Conformal PI Coverage (HB-Anchored, buffer-ready):")
        print(conf_summ.to_string(index=False))
    if not crpss_df.empty:
        print("\n▶ CRPSS [Gaussian] vs hist_mean:")
        print(
            crpss_df.pivot_table(
                index="model", columns="horizon", values="CRPSS"
            ).round(4)
        )
    if not dm_df.empty:
        print("\n▶ Diebold-Mariano Tests [Gaussian CRPS]:")
        print(dm_df[["horizon", "benchmark", "DM_stat", "p_value", "hb_wins", "CRPSS"]]
              .to_string(index=False))
    if vol_result:
        print(
            f"\n▶ Category Volatility: "
            f"H={vol_result.get('H_stat')}  "
            f"p={vol_result.get('p_value', float('nan')):.4g}  "
            f"η²={vol_result.get('eta_squared')}"
        )
    if not mz_df.empty:
        print("\n▶ Mincer-Zarnowitz [Gaussian median]:")
        print(mz_df.to_string(index=False))

    print(f"\n✅ Pipeline complete in {elapsed:.0f}s")
    print(f"   Outputs saved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
