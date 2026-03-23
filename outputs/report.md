#  Pipeline  — Results Report
Generated: 2026-03-23 15:50:42
Elapsed: 343s

##  Core Change: Dual-Distribution Architecture
CRPS Track:     Gaussian N(0,σ_innov)  → accuracy scoring
Coverage Track: Student-t(ν≥8) × inflate → model PI width
Conformal Track: q_conf from Gaussian errors → dist.-free guarantee
FIX-6: T_DOF_MIN raised 4→8
FIX-7: Gaussian track has NO sigma inflation (raw σ_innov)
FIX-8: MZ unbiasedness test uses Gaussian median

## Log-CRPS [Gaussian track]
horizon             1       3       7
model                                
complete_pool  1.0357  1.0620  1.0376
ets            0.1392  0.1536  0.2677
hb_anchored    0.1268  0.1313  0.2792
hb_mean        0.1554  0.1625  0.2658
hist_mean      0.1501  0.1624  0.2368
naive          0.0950  0.1287  0.3055
no_pool        0.1591  0.1637  0.2713

## Model PI Coverage [Student-t track]
horizon            1      3      7
model                             
complete_pool  0.660  0.660  0.651
ets            0.633  0.780  0.385
hb_anchored    0.840  0.850  0.538
hb_mean        0.706  0.845  0.571
hist_mean      0.648  0.533  0.240
naive          0.840  0.808  0.394
no_pool        0.384  0.630  0.396

## Conformal PI Coverage
 horizon  n_forecasts  cov_model_PI  cov_conf_PI  gain_pp  target
       1         2163        0.8410       0.9075     6.66     0.9
       3         1541        0.8494       0.9215     7.20     0.9
       7          308        0.5130       0.9643    45.13     0.9

## CRPSS [Gaussian]
        model  horizon   CRPSS  mean_log_crps
        naive        1  0.3671         0.0950
  hb_anchored        1  0.1549         0.1268
          ets        1  0.0725         0.1392
    hist_mean        1  0.0000         0.1501
      hb_mean        1 -0.0356         0.1554
      no_pool        1 -0.0597         0.1591
complete_pool        1 -5.9000         1.0357
        naive        3  0.2076         0.1287
  hb_anchored        3  0.1918         0.1313
          ets        3  0.0542         0.1536
    hist_mean        3  0.0000         0.1624
      hb_mean        3 -0.0004         0.1625
      no_pool        3 -0.0082         0.1637
complete_pool        3 -5.5385         1.0620
    hist_mean        7  0.0000         0.2368
      hb_mean        7 -0.1223         0.2658
          ets        7 -0.1304         0.2677
      no_pool        7 -0.1456         0.2713
  hb_anchored        7 -0.1789         0.2792
        naive        7 -0.2902         0.3055
complete_pool        7 -3.3816         1.0376

## DM Tests
 horizon     benchmark  n_pairs  DM_stat  p_value  hb_wins  mean_crps_hb  mean_crps_bm   CRPSS
       1         naive     2193   -2.765   0.0057        0        0.1268        0.0950 -0.3352
       1           ets     2193   -2.283   0.0225        1        0.1268        0.1392  0.0889
       1     hist_mean     2193   -2.327   0.0201        1        0.1268        0.1501  0.1549
       1       no_pool     2193   -5.215   0.0000        1        0.1268        0.1591  0.2025
       1 complete_pool     2193  -26.373   0.0000        1        0.1268        1.0357  0.8775
       1       hb_mean     2193   -3.787   0.0002        1        0.1268        0.1554  0.1840
       3         naive     1571   -2.904   0.0037        0        0.1313        0.1287 -0.0198
       3           ets     1571   -2.604   0.0093        1        0.1313        0.1536  0.1455
       3     hist_mean     1571   -1.182   0.2374        1        0.1313        0.1624  0.1918
       3       no_pool     1571   -1.652   0.0988        1        0.1313        0.1637  0.1984
       3 complete_pool     1571  -13.516   0.0000        1        0.1313        1.0620  0.8764
       3       hb_mean     1571   -2.003   0.0454        1        0.1313        0.1625  0.1921
       7         naive      338   -2.450   0.0148        1        0.2792        0.3055  0.0863
       7           ets      338    1.689   0.0922        0        0.2792        0.2677 -0.0429
       7     hist_mean      338    3.141   0.0018        0        0.2792        0.2368 -0.1789
       7       no_pool      338    0.922   0.3573        0        0.2792        0.2713 -0.0290
       7 complete_pool      338   -4.742   0.0000        1        0.2792        1.0376  0.7309
       7       hb_mean      338    1.642   0.1016        0        0.2792        0.2658 -0.0504

## MZ Tests
        model  horizon   alpha   beta     R2  unbiased
  hb_anchored        1 -0.0582 1.0089 0.9062         1
  hb_anchored        3 -0.0352 1.0079 0.9244         1
  hb_anchored        7  0.8894 0.9189 0.8602         0
      hb_mean        1 -0.0237 1.0051 0.9024         1
      hb_mean        3 -0.0815 1.0130 0.9087         1
      hb_mean        7 -0.1735 1.0270 0.8641         0
        naive        1  0.5438 0.9483 0.8977         0
        naive        3  0.5385 0.9512 0.9058         0
        naive        7  1.9787 0.8244 0.6811         0
          ets        1 -0.0399 1.0066 0.9071         1
          ets        3 -0.1025 1.0148 0.9164         0
          ets        7 -0.0580 1.0151 0.8736         1
    hist_mean        1  0.0543 0.9965 0.9070         1
    hist_mean        3 -0.0369 1.0072 0.9231         1
    hist_mean        7 -0.0988 1.0169 0.8992         1
      no_pool        1  0.0590 0.9960 0.9075         1
      no_pool        3 -0.0340 1.0069 0.9229         1
      no_pool        7 -0.0720 1.0141 0.8995         1
complete_pool        1  7.3793 0.3407 0.0113         0
complete_pool        3  6.9494 0.3917 0.0157         0
complete_pool        7  3.5974 0.7580 0.0533         0