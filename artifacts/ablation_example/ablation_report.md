# Feature Ablation Study Report

**Sprint 1 Bootstrap — Zero-Cost Implementation**

Testing incremental feature addition to measure OOS contribution.

## Summary Table

| Feature Set | # Features | Accuracy | Precision | F1 | Coverage | Train Time |
|-------------|------------|----------|-----------|----|---------|-----------|
| base | 20 | 0.755 | 0.769 | 0.760 | 100.0% | 0.2s |
| base+micro | 35 | 0.873 | 0.887 | 0.875 | 100.0% | 0.3s |
| base+micro+mtf | 50 | 0.877 | 0.883 | 0.880 | 100.0% | 0.4s |
| base+micro+mtf+smc | 65 | 0.898 | 0.900 | 0.901 | 100.0% | 0.5s |
| base+micro+mtf+smc+intermarket | 80 | 0.922 | 0.920 | 0.925 | 100.0% | 0.6s |

## Incremental Gains

Improvement from adding each feature family:

| Added Family | Δ Accuracy | Δ Precision | Δ F1 |
|--------------|------------|-------------|------|
| micro | +0.118 | +0.117 | +0.116 |
| mtf | +0.004 | -0.004 | +0.005 |
| smc | +0.021 | +0.017 | +0.021 |
| intermarket | +0.024 | +0.020 | +0.024 |

## Top Features Per Configuration

### base (Top 5)

1. `feat_2`: 0.1044
2. `feat_15`: 0.0589
3. `feat_11`: 0.0563
4. `feat_5`: 0.0559
5. `feat_1`: 0.0522

### base+micro (Top 5)

1. `feat_33`: 0.0690
2. `feat_2`: 0.0561
3. `feat_11`: 0.0421
4. `feat_20`: 0.0402
5. `feat_32`: 0.0391

### base+micro+mtf (Top 5)

1. `feat_33`: 0.0533
2. `feat_2`: 0.0447
3. `feat_43`: 0.0422
4. `feat_11`: 0.0328
5. `feat_42`: 0.0310

### base+micro+mtf+smc (Top 5)

1. `feat_43`: 0.0380
2. `feat_2`: 0.0379
3. `feat_33`: 0.0349
4. `feat_50`: 0.0256
5. `feat_64`: 0.0245

### base+micro+mtf+smc+intermarket (Top 5)

1. `feat_43`: 0.0306
2. `feat_33`: 0.0295
3. `feat_2`: 0.0273
4. `feat_68`: 0.0243
5. `feat_69`: 0.0225


## Recommendations

**Best configuration:** base+micro+mtf+smc+intermarket

- Features: 80
- F1 Score: 0.925
- Precision: 0.920

✅ All feature families improved OOS performance.

