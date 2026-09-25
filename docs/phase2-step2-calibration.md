# Phase II, Step 2: Confidence calibration (Jev)

**Question:** does Jev's reported confidence correspond to actual correctness?

**Status:** results generated; interpretation pending review. This document reports numbers with their sample counts. It does not conclude that Jev is or is not well calibrated.

- **Source:** `data/results/phase2/baseline/canonical_results.parquet` (frozen canonical baseline, 6,600 rows; 6,580 labeled and used).
- **Outputs:** `data/results/phase2/calibration/` (`calibration_bins.csv`, `calibration_summary.csv`, `calibration_meta.json`).
- **Code:** `src/phase2/calibration.py`; run with `uv run python src/phase2/run_calibration.py`.

## What is measured

- **Confidence** is the probability Jev assigned to its *chosen* label, stored to 2 decimals. The full distribution over candidates is not stored, so nothing here uses or infers it.
- **Bins:** 10 equal-width bins, `[0.0, 0.1)` ... `[0.9, 1.0]` (last bin closed).
- **ECE** = sum over bins of (n_bin / N) x |accuracy_bin - mean_confidence_bin|. Empty bins contribute 0.
- **Brier** is the *correctness* Brier score, `mean((confidence - correct)^2)` on chosen-label confidence. It is **not** a multiclass Brier score.
- Pooled levels count a re-measured example once (latest row, the same outcome-independent rule as Step 1). Per-run levels use each run as is.

## Jev headline results (n always shown)

| Dataset | n | Distinct examples | Accuracy | Mean confidence | ECE | Brier (correctness) | Mean conf. when correct | Mean conf. when incorrect |
|---|---|---|---|---|---|---|---|---|
| Bitext | 1000 | 1000 | 0.941 | 0.951 | 0.015 | 0.046 | 0.963 | 0.761 |
| MASSIVE (pooled rows) | 1200 | 198 | 0.799 | 0.883 | 0.086 | 0.135 | 0.922 | 0.725 |

Accuracy and calibration are different properties; neither implies the other.

### Jev by experiment (MASSIVE pooled rows split into their workloads)

| Experiment | n | Distinct examples | Accuracy | Mean conf. | ECE | Brier | Sparse bins (n < 30) |
|---|---|---|---|---|---|---|---|
| bitext_hard_choice | 1000 | 1000 | 0.941 | 0.951 | 0.015 | 0.046 | 4 |
| choice_scaling_k5 | 100 | 100 | 0.880 | 0.938 | 0.088 | 0.090 | 5 |
| choice_scaling_k10 | 100 | 100 | 0.850 | 0.931 | 0.088 | 0.097 | 5 |
| choice_scaling_k25 | 100 | 100 | 0.770 | 0.888 | 0.145 | 0.163 | 6 |
| choice_scaling_k60 | 100 | 100 | 0.730 | 0.859 | 0.152 | 0.176 | 6 |
| multilingual | 800 | 100 | 0.795 | 0.872 | 0.082 | 0.137 | 2 |

The MASSIVE row above pools different workloads (choice scaling and multilingual); the per-experiment rows are the non-pooled view. ECE from 100-sample groups is noisy.

## Caveats that must accompany any claim

1. **Sparse bins.** Bins with fewer than 30 predictions are unreliable; their accuracy estimates are very noisy.
   - Jev / Bitext (pooled): `[0.3,0.4)` n=1, `[0.4,0.5)` n=9, `[0.5,0.6)` n=25, `[0.6,0.7)` n=22.
   - Jev / MASSIVE (pooled): `[0.2,0.3)` n=6, `[0.3,0.4)` n=19.
   - Empty bins (e.g. all bins below 0.3 for Bitext) are kept in the tables with `sample_count = 0`.
2. **MASSIVE rows are not independent.** The 8 locales are aligned translations of the same utterances, so 1,200 pooled rows come from only 198 distinct example IDs (the multilingual experiment alone is 800 rows from 100 IDs). Effective sample size is far smaller than n, and any uncertainty derived from n=1200 would be overstated.
3. **Mass at confidence = 1.0.** Across the pooled canonical data, 2,925 labeled predictions have confidence exactly 1.0, and **133 of them are wrong**. That 133 covers *both providers* and *all 15 canonical runs without deduplicating re-measured examples*. The Jev-only breakdown:

   | Scope | Rows at conf = 1.0 | Wrong |
   |---|---|---|
   | Both providers, all canonical runs (the 133) | 2,925 | 133 |
   | Jev only, all canonical runs | 2,200 | 36 (Bitext 10 of 1,429; MASSIVE 26 of 771) |
   | Jev only, deduplicated pooled view (matches the tables above) | 1,096 | 23 (Bitext 5 of 610; MASSIVE 18 of 486) |

4. **Confidence resolution** is 2 decimals, which limits bin granularity.
5. **Single measurement per example** (after deduplication). Repeat-call disagreement is recorded separately in the Step 1 dedup report (8 of 400 repeated identities disagreed).

## Reference provider

`gpt-4o-mini` rows remain in the generated CSVs for provenance. It is a reference/fallback provider only; its calibration results are not used for Phase II conclusions and its confidence behavior has not been investigated.

## Not concluded

Jev is not called well or poorly calibrated here. That judgment waits on review of the reliability bins together with the sample counts above.
