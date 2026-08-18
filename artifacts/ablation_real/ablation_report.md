# Feature Ablation Study Report

**Sprint 1 Bootstrap — Zero-Cost Implementation**

Testing incremental feature addition to measure OOS contribution.

## Summary Table

| Feature Set | # Features | Accuracy | Precision | F1 | Coverage | Train Time |
|-------------|------------|----------|-----------|----|---------|-----------|
| base | 26 | 0.617 | 0.500 | 0.441 | 100.0% | 0.1s |
| base+micro | 30 | 0.644 | 0.545 | 0.482 | 100.0% | 0.1s |
| base+micro+mtf | 30 | 0.644 | 0.545 | 0.482 | 100.0% | 0.1s |
| base+micro+mtf+smc | 36 | 0.633 | 0.526 | 0.458 | 100.0% | 0.1s |
| base+micro+mtf+smc+intermarket | 54 | 0.632 | 0.524 | 0.462 | 100.0% | 0.1s |

## Incremental Gains

Improvement from adding each feature family:

| Added Family | Δ Accuracy | Δ Precision | Δ F1 |
|--------------|------------|-------------|------|
| micro | +0.027 | +0.045 | +0.041 |
| mtf | +0.000 | +0.000 | +0.000 |
| smc | -0.012 | -0.019 | -0.024 |
| intermarket | -0.001 | -0.002 | +0.004 |

## Top Features Per Configuration

### base (Top 5)

1. `session_ny`: 0.0792
2. `dist_session_low`: 0.0700
3. `dist_session_high`: 0.0586
4. `session_london`: 0.0495
5. `session_asia`: 0.0495

### base+micro (Top 5)

1. `session_asia`: 0.0730
2. `session_ny`: 0.0677
3. `dist_session_low`: 0.0617
4. `dist_session_high`: 0.0558
5. `session_london`: 0.0462

### base+micro+mtf (Top 5)

1. `session_asia`: 0.0730
2. `session_ny`: 0.0677
3. `dist_session_low`: 0.0617
4. `dist_session_high`: 0.0558
5. `session_london`: 0.0462

### base+micro+mtf+smc (Top 5)

1. `dist_session_low`: 0.0558
2. `dist_session_high`: 0.0515
3. `session_ny`: 0.0475
4. `bos_bear_recent`: 0.0374
5. `session_london`: 0.0366

### base+micro+mtf+smc+intermarket (Top 5)

1. `session_ny`: 0.0573
2. `session_asia`: 0.0517
3. `dist_session_low`: 0.0516
4. `dist_session_high`: 0.0508
5. `session_range_pct`: 0.0406


## Recommendations

**Best configuration:** base+micro

- Features: 30
- F1 Score: 0.482
- Precision: 0.545

⚠️ **Warning:** These families reduced F1 score: smc

Consider removing these features or investigating for leakage/overfitting.

