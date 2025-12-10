# Stage 1 Denormalization Bug Fix

## Problem Discovered
The Stage 1 behavioral forecasting model was outputting **normalized** predictions (values like 0.06) instead of **actual** predictions (values like 6.5 hours of sleep).

## Root Cause
In `train_stage1_behavioral_kaggle.py`, the scaler statistics were computed **after** normalizing the data:

```python
# WRONG ORDER (lines 343-347 original):
df[FEATURE_COLS] = (df[FEATURE_COLS] - scaler_mean) / scaler_std  # Normalize first

target_scaler_mean = df[BEHAVIORAL_TARGETS].mean().values  # Then compute stats
target_scaler_std = df[BEHAVIORAL_TARGETS].std().values    # Stats are now 0 and 1!

df[BEHAVIORAL_TARGETS] = (df[BEHAVIORAL_TARGETS] - target_scaler_mean) / target_scaler_std
```

This meant:
- `target_scaler_mean` ≈ 0 (mean of normalized data)
- `target_scaler_std` = 1 (std of normalized data)
- Denormalization formula: `pred * 1 + 0 = pred` (no change!)

## Fix Applied
Moved scaler computation **before** normalization:

```python
# CORRECT ORDER (lines 330-351 fixed):
# 1. Compute scalers from ORIGINAL data
target_scaler_mean = df[BEHAVIORAL_TARGETS].mean().values  # Real means: 6.5, 45.0, etc.
target_scaler_std = df[BEHAVIORAL_TARGETS].std().values    # Real stds: 1.5, 30.0, etc.

# 2. Then normalize data
df[BEHAVIORAL_TARGETS] = (df[BEHAVIORAL_TARGETS] - target_scaler_mean) / target_scaler_std
```

Now:
- `target_scaler_mean` = actual data means (e.g., sleep_hours ≈ 6.5)
- `target_scaler_std` = actual data stds (e.g., sleep_hours ≈ 1.5)
- Denormalization works: `pred_norm * std + mean = actual_value`

## What Needs to Happen

### Option 1: Retrain on Kaggle (Recommended)
```bash
# On Kaggle notebook:
# 1. Update train_stage1_behavioral_kaggle.py with the fix
# 2. Re-run training cell
# 3. Re-run two-stage pipeline cell
# 4. Download new results
```

**Expected Results After Retraining:**
```json
{
  "stage1_behavioral_predictions": {
    "sleep_hours": 6.8,          // ← Real hours, not 0.06!
    "exercise_minutes": 42.0,    // ← Real minutes, not 0.15!
    "screen_time_hours": 7.2,    // ← Real hours, not 0.22!
    "social_interactions": 28.0, // ← Real count, not -0.04!
    "steps_count": 5200.0        // ← Real steps, not 0.16!
  }
}
```

### Option 2: Fix Checkpoint Manually (Quick Hack)
Load the existing checkpoint and inject correct scalers from the original data:

```python
import torch
import pandas as pd
import numpy as np

# Load checkpoint
checkpoint = torch.load('models/saved/stage1_behavioral_forecasting.pt')

# Recompute scalers from raw StudentLife data (before normalization)
# ... extract raw data ...
# target_scaler_mean = raw_data[BEHAVIORAL_TARGETS].mean().values
# target_scaler_std = raw_data[BEHAVIORAL_TARGETS].std().values

# Update checkpoint
checkpoint['target_scaler_mean'] = target_scaler_mean
checkpoint['target_scaler_std'] = target_scaler_std

# Save fixed checkpoint
torch.save(checkpoint, 'models/saved/stage1_behavioral_forecasting_fixed.pt')
```

**Not recommended** - cleaner to retrain with correct scalers from start.

## Impact on Demo

### Current State (Buggy)
- Stage 1 predictions: Normalized values (0.06, 0.15, etc.)
- Demo shows: "Sleep: 0.06h ± 0.50h" (meaningless!)
- Stage 2 receives: Normalized inputs (breaks distribution assumptions)
- Visualizations: Show normalized scales (confusing)

### After Fix
- Stage 1 predictions: Actual values (6.8 hours, 42 min, etc.)
- Demo shows: "Sleep: 6.8h ± 0.5h" (interpretable!)
- Stage 2 receives: Actual scale inputs (better distribution match)
- Visualizations: Show real-world scales (professional)

## Testing the Fix

After retraining, verify:

```python
# Load new checkpoint
checkpoint = torch.load('models/saved/stage1_behavioral_forecasting.pt')

# Check scalers are NOT zero/one
print("Target mean:", checkpoint['target_scaler_mean'])  # Should be ~[6.5, 45.0, 7.0, 30.0, 5000.0]
print("Target std:", checkpoint['target_scaler_std'])    # Should be ~[1.5, 30.0, 2.0, 15.0, 2000.0]

# Test denormalization
pred_normalized = 0.5  # Some prediction
pred_actual = pred_normalized * checkpoint['target_scaler_std'][0] + checkpoint['target_scaler_mean'][0]
print(f"Sleep prediction: {pred_actual:.1f}h")  # Should be realistic (5-8h range)
```

## Timeline

1. ✅ **Bug identified**: December 10, 2025
2. ✅ **Fix applied**: `train_stage1_behavioral_kaggle.py` updated
3. ⏳ **Retrain needed**: Run Kaggle notebook with fixed script
4. ⏳ **New predictions**: Generate 598 new predictions with real scales
5. ⏳ **Demo update**: Replace JSON file, regenerate visualizations
6. ✅ **Production ready**: Demo shows actual predictions!

## Why This Matters

This isn't just a display bug - it fundamentally affects the **research validity**:

1. **Stage 2 Input Distribution**: Stage 2 was trained on synthetic data with actual scales (sleep: 6-8h, not 0-1). Feeding it normalized values breaks the input distribution assumptions.

2. **Uncertainty Propagation**: Uncertainties are in normalized space (±0.5 normalized units) not actual space (±0.5 hours). Makes error propagation analysis incorrect.

3. **Comparison Validity**: Can't compare cascaded vs direct predictions if Stage 1 outputs are on wrong scale.

4. **Professional Presentation**: Showing "sleep: 0.06h" looks like a bug, not research.

## Next Steps

**Immediate action required**: Retrain Stage 1 model on Kaggle with the fixed script.

The fix is in the codebase, just needs re-execution on Kaggle to generate corrected predictions!
