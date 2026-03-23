# Hierarchical Bayesian Probabilistic Forecasting Under Extreme Data Scarcity

> **Cross-Sectional Information Pooling with Dual-Distribution Architecture and Conformal Calibration for Sparse Agricultural Price Series**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: KAMIS](https://img.shields.io/badge/Data-KAMIS%20Public%20API-green)](https://www.kamis.or.kr/)

---

## Overview

This repository contains the full replication code and outputs for a study on **probabilistic forecasting under extreme data scarcity**, using Korean agricultural wholesale and retail price data from the [Korea Agricultural Marketing Information Service (KAMIS)](https://www.kamis.or.kr/).

The central methodological contribution is a **Dual-Distribution Architecture** (v9) that separates the scoring objective from the coverage objective in probabilistic forecasting:

| Track | Distribution | Purpose | Metrics |
|-------|-------------|---------|---------|
| **CRPS Track** | Gaussian N(μ, σ_innov) | Accuracy scoring | Log-CRPS, MAE, RMSE, DM test, CRPSS, Mincer-Zarnowitz |
| **Coverage Track** | Student-t (ν ≥ 8) × inflate | Interval calibration | 90% PI coverage, Winkler score |
| **Conformal Track** | Distribution-free | Theoretical guarantee | Coverage guarantee ≥ 90% |

### Key Results (Rolling Out-of-Sample Evaluation, 2024–2025)

| Horizon | HB-Anchored Log-CRPS | CRPSS vs Climatology | Conformal PI Coverage |
|---------|---------------------|---------------------|----------------------|
| h = 1d  | 0.127 | **+15.5%** | 90.8% |
| h = 3d  | 0.131 | **+19.2%** | 92.2% |
| h = 7d  | 0.279 | −17.9%     | **96.4%** |

> At h=7, conformal calibration recovers 90%+ coverage (from 51% model PI) by accumulating distribution-free prediction errors online.

---

## Architecture

```
y_train (n sparse observations)
    ↓
HB Level: ETS + James-Stein shrinkage + dampened trend
    ↓
Innovation variance σ²_innov
    ├─ [Gaussian N(0, σ_innov)] ──────────────────→  CRPS Track
    │      log_crps, MAE, RMSE, DM test, CRPSS       (accuracy)
    │      Mincer-Zarnowitz, conformal buffer anchor
    │
    ├─ [Student-t(ν≥8) × inflate_h] ─────────────→  Coverage Track
    │      90% model PI, Winkler score                (calibration)
    │
    └─ [Conformal buffer on Gaussian errors] ──────→  Conformal Track
           q_conf = Quantile(|Gaussian errors|)       (guarantee)
           PI_conf = [ŷ_gauss ± q_conf]
```

---

## Repository Structure

```
hb-sparse-agri-forecasting/
│
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
├── .gitignore
│
├── pipeline.py                   # Main replication script (full pipeline)
│
├── outputs/                         # Pre-computed results (uploaded)
│   ├── figures/
│   │   ├── fig1_dual_architecture.png
│   │   ├── fig2_sparsity_experiment.png
│   │   ├── fig3_dual_architecture_diagram.png
│   │   ├── fig4_category_volatility.png
│   │   ├── fig5_crpss_heatmap.png
│   │   ├── fig6_dm_heatmap.png
│   │   ├── fig7_pooling_gain.png
│   │   └── fig8_conformal_gain.png
│   │
│   ├── tables/
│   │   ├── table1_accuracy.csv           # Log-CRPS, Coverage, Winkler, MAE, RMSE by model×horizon
│   │   ├── table2_sparsity_h1.csv        # Sparsity simulation: h=1
│   │   ├── table2_sparsity_h3.csv        # Sparsity simulation: h=3
│   │   ├── table2_sparsity_h7.csv        # Sparsity simulation: h=7
│   │   ├── table3_coverage.csv           # Model PI coverage by model×horizon
│   │   ├── dm_tests.csv                  # Diebold-Mariano test results
│   │   ├── crpss.csv                     # CRPSS vs climatology
│   │   ├── mz.csv                        # Mincer-Zarnowitz unbiasedness tests
│   │   ├── conformal.csv                 # Conformal PI coverage summary
│   │   └── category_volatility.csv       # Kruskal-Wallis volatility heterogeneity
│   │
│   └── ijf_report.md                     # Full numerical results report
```

---

## Data

Data are fetched live from the **KAMIS public API** (Korea Agro-Fisheries & Food Trade Corporation).

- **Training period**: January 1, 2024 – December 31, 2024
- **Test period**: January 1, 2025 – present (rolling)
- **Coverage**: 97 agricultural items across retail and wholesale markets nationwide
- **After cleaning**: 11,430 observations, 2024-01-01 to 2025-01-13

### API Access

The KAMIS API is publicly accessible. A service key is required and can be obtained at [data.go.kr](https://www.data.go.kr/). Set your key as an environment variable:

```bash
export DATA_GO_KR_KEY="your_service_key_here"
```

The pipeline will fall back to the bundled key for read-only reproducibility (subject to API rate limits).

---

## Installation

### Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

### Setup

```bash
git clone https://github.com/LEEYJ1021/hb-sparse-agri-forecasting.git
cd hb-sparse-agri-forecasting
pip install -r requirements.txt
```

---

## Replication

### Full Pipeline (all stages)

```bash
python pipeline.py
```

This runs all 7 stages sequentially:

| Stage | Description | Approx. Time |
|-------|-------------|-------------|
| 1. Data | Fetch from KAMIS API | ~30s |
| 2. Preparation | Clean, log-transform, build hierarchy | ~10s |
| 3. Rolling OOS | 338 item-market pairs, horizons h=1,3,7 | ~200s |
| 4. Sparsity Simulation | n∈{3,5,7}, 20 reps per item | ~80s |
| 5. Statistical Tests | DM, Kruskal-Wallis, Mincer-Zarnowitz | ~5s |
| 6. Tables | Build summary CSVs | ~5s |
| 7. Figures | Generate all 8 figures | ~15s |

**Total runtime**: ~343 seconds (5–6 minutes)

### Output Directory

By default, outputs are saved to `./outputs/`. Modify `OUT` in `pipeline.py` if needed:

```python
OUT = Path("./outputs")   # Change this to your preferred path
```

---

## Methods

### Hierarchical Bayesian Model

Three-level model for log-transformed prices y_ijt (item i, market j, time t):

- **Level 1** (Observation): y_ijt | μ_ij, σ² ~ N(μ_ij, σ²)
- **Level 2** (Item-market): μ_ij | μ_c, τ²_c ~ N(μ_c, τ²_c)
- **Level 3** (Hyperpriors): μ_c ~ N(μ₀, τ₀²), τ²_c ~ InvGamma(α, β)

### James-Stein Shrinkage

Posterior mean shrinks item estimates toward the category mean:

```
E[μ_ij | data] = (1 - λ_ij) · ȳ_ij + λ_ij · μ_c
λ_ij = σ² / (n_ij · τ²_c + σ²)
```

As n → 0: λ → 1 (complete pooling). As n → ∞: λ → 0 (no pooling).

### Adaptive Estimation

| Data regime | Method | % of sample |
|-------------|--------|------------|
| n ≥ 20 | Dynamic Linear Model (Kalman filter) | 0% (current window) |
| 10 ≤ n < 20 | Full Bayesian MCMC (Gibbs sampler) | ~27% |
| 3 ≤ n < 10 | Empirical Bayes (James-Stein) | ~73% |

### Dual-Distribution Architecture (v9 Core Contribution)

**Problem**: A single Student-t distribution cannot simultaneously optimize CRPS (accuracy) and PI coverage, because fat tails that widen intervals also accumulate CRPS penalty when the true DGP is not heavy-tailed.

**Solution**: Use two separate distributions for two separate objectives.

**CRPS Track** (Gaussian):
```python
log_samples_g = N(level_h, σ_innov)    # raw σ, no inflation
```
Gaussian minimizes CRPS when the conditional mean is well-estimated.

**Coverage Track** (Student-t):
```python
log_samples_t = t(ν, level_h, σ_innov × inflate_h)
# ν = clip(n-1, 8, 30)
# inflate_h ∈ {1.0, 1.3, 1.8} for h ∈ {1, 3, 7}
```
Fat tails widen prediction intervals without incurring CRPS penalty.

### Online Conformal Calibration

An online split conformal buffer accumulates absolute Gaussian median errors:

```
q_conf(h) = Quantile_{⌈(m+1)(1-α)/m⌉}(|y_t - ŷ_gauss,t|)  for m ≥ 30
PI_conf = [ŷ_gauss ± q_conf]
```

This provides distribution-free coverage guarantees without any parametric assumptions.

### Benchmarks

| Model | Description |
|-------|-------------|
| `naive` | Random walk (last observed value) |
| `ets` | Exponential smoothing (ETS) |
| `hist_mean` | Historical mean (climatological baseline) |
| `no_pool` | Item-specific estimation, no cross-sectional pooling |
| `complete_pool` | All items share category mean and variance |
| `hb_mean` | Hierarchical Bayes with mean-reversion level |
| `hb_anchored` | **Proposed**: HB with ETS level + JS shrinkage + dampened trend |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Log-CRPS | Continuous Ranked Probability Score on log scale (Gneiting & Raftery, 2007) |
| CRPSS | CRPS Skill Score relative to climatological baseline (hist_mean) |
| DM test | Diebold-Mariano test for equal predictive accuracy (Diebold & Mariano, 1995) |
| Coverage | 90% prediction interval empirical coverage |
| Winkler score | Interval sharpness penalized for misses |
| Mincer-Zarnowitz | Forecast unbiasedness test: y_t = α + β·ŷ_t + ε, H₀: α=0, β=1 |

---

## Key Numerical Results

### Log-CRPS (Gaussian Track, lower = better)

| Model | h=1 | h=3 | h=7 |
|-------|-----|-----|-----|
| **HB-Anchored** | **0.127** | **0.131** | 0.279 |
| ETS | 0.139 | 0.154 | 0.268 |
| Hist. Mean | 0.150 | 0.162 | **0.237** |
| Naïve | 0.095 | 0.129 | 0.306 |
| No-Pool | 0.159 | 0.164 | 0.271 |
| Complete-Pool | 1.036 | 1.062 | 1.038 |

### Conformal PI Coverage (HB-Anchored, buffer-ready periods)

| Horizon | n_forecasts | Model PI (Student-t) | Conformal PI | Gain |
|---------|------------|---------------------|--------------|------|
| h = 1d | 2,163 | 84.1% | **90.8%** | +6.7pp |
| h = 3d | 1,541 | 84.9% | **92.2%** | +7.2pp |
| h = 7d | 308  | 51.3% | **96.4%** | +45.1pp |

### Diebold-Mariano Tests (HB-Anchored vs. Benchmarks)

| Benchmark | h=1 p-value | h=3 p-value | h=7 p-value |
|-----------|------------|------------|------------|
| ETS | 0.022 ✅ | 0.009 ✅ | 0.092 |
| Hist. Mean | 0.020 ✅ | 0.237 | 0.002* |
| No-Pool | <0.001 ✅ | 0.099 | 0.357 |
| Naïve | 0.006* | 0.004* | 0.015 ✅ |
| Complete-Pool | <0.001 ✅ | <0.001 ✅ | <0.001 ✅ |

✅ = HB-Anchored significantly better (p < 0.10). * = HB-Anchored loses.

### Category Volatility Heterogeneity

Kruskal-Wallis test across 6 categories: **H = 22.98, p < 0.001, η² = 0.214**

| Category | Median CV | n items |
|----------|-----------|---------|
| Staple Crops | 1.09 | 8 |
| Industrial Crops | 0.93 | 8 |
| Vegetables | 0.80 | 21 |
| Fruits | 0.62 | 13 |
| Seafood | 0.36 | 22 |
| Processed Foods | 0.12 | 8 |

---

## Figures

| Figure | Description |
|--------|-------------|
| `fig1_dual_architecture.png` | Main results: CRPS (Row 1), Model PI coverage (Row 2), Conformal PI coverage (Row 3) across h=1,3,7 |
| `fig2_sparsity_experiment.png` | Log-CRPS vs. training window n ∈ {3,5,7} by horizon |
| `fig3_dual_architecture_diagram.png` | Architecture diagram: CRPS evolution v8→v9, coverage comparison, flowchart |
| `fig4_category_volatility.png` | Violin plots and bar chart of CV heterogeneity across categories |
| `fig5_crpss_heatmap.png` | CRPSS (Gaussian) vs. climatological baseline heatmap |
| `fig6_dm_heatmap.png` | Diebold-Mariano p-value heatmap (HB-Anchored vs. benchmarks) |
| `fig7_pooling_gain.png` | Average pooling gain and win-rate vs. No-Pooling across n and h |
| `fig8_conformal_gain.png` | Conformal calibration effect: Model PI vs. Conformal PI per horizon |

---

## Configuration

Key parameters in `pipeline.py`:

```python
# Forecasting horizons
HORIZONS = [1, 3, 7]               # days ahead

# Dual-distribution parameters
T_DOF_MIN = 8                       # min Student-t dof (coverage track)
COVERAGE_SIGMA_INFLATE = {1: 1.0, 3: 1.3, 7: 1.8}  # sigma inflation by horizon

# Conformal calibration
CONFORMAL_MIN_BUFFER = 30           # min errors before conformal PI activates
TARGET_COVERAGE = 0.90              # target nominal coverage

# Variance control
INNOV_N0 = 2                        # prior sample size for variance blending
SIGMA_FLOOR_MULT = 0.5              # sqrt(h) floor multiplier
VAR_CAP_RATIO = 6                   # cap = floor × ratio

# Sparsity simulation
SPARSITY_LEVELS = [3, 5, 7]         # training window sizes
N_SIM_REPS = 20                     # Monte Carlo repetitions per item
```

---

## Reproducibility Notes

- **Random seed**: `np.random.seed(42)` is set globally. Monte Carlo results are fully reproducible given the same API data pull.
- **API variability**: KAMIS API data may be updated retroactively. For exact replication of the reported results, use the pre-computed CSVs in `outputs/tables/`.
- **Runtime**: ~343 seconds on a standard laptop (Intel Core i7, 16GB RAM). The rolling OOS loop (~200s) is the bottleneck.
- **Python version**: Tested on Python 3.9 and 3.11.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Data sourced from KAMIS (Korea Agricultural Marketing Information Service) is subject to the [Korea Open Government License (KOGL) Type 1](https://www.kamis.or.kr/).

---

## Contact

**Yong-Jae Lee**  
GitHub: [@LEEYJ1021](https://github.com/LEEYJ1021)

For questions about the methodology or replication, please open an [Issue](https://github.com/LEEYJ1021/hb-sparse-agri-forecasting/issues).
