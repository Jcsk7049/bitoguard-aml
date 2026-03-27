# BitoGuard Pipeline Fix Summary

## Issue Resolved
The pipeline was failing at the XAI stage with two errors:
1. **Model Loading Error**: XGBoost couldn't find `model.json` when running `--start-from xai`
2. **JSON Serialization Error**: NumPy float32 values weren't JSON serializable

## Root Causes

### Issue 1: Model Path Resolution
- When running `--start-from xai`, the pipeline skipped earlier stages
- The model path wasn't being properly resolved relative to the current working directory
- XGBoost's `load_model()` was failing with "No such file or directory"

### Issue 2: JSON Serialization
- The SHAP contribution values were stored as NumPy float32 types
- Python's standard `json.dump()` doesn't support NumPy numeric types
- This caused a `TypeError: Object of type float32 is not JSON serializable`

## Solutions Implemented

### Fix 1: Improved Model Path Resolution (main_pipeline.py)
Added robust path resolution logic in the XAI stage:
```python
# Ensure model_path is absolute or relative to current directory
if not os.path.isabs(model_path) and not os.path.exists(model_path):
    alt_path = os.path.join(os.getcwd(), model_path)
    if os.path.exists(alt_path):
        model_path = alt_path
```

Also enhanced the `stage_xai()` function with:
- Explicit path existence checking before loading
- Better error messages for debugging
- Fallback to default `model.json` if path resolution fails

### Fix 2: Custom JSON Encoder (xai_bedrock.py)
Added a `NumpyEncoder` class to handle NumPy numeric types:
```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return super().default(obj)
```

## Test Results

✅ **Pipeline Execution**: Successfully completed `--start-from xai`
- Model loaded: ✓
- Predictions generated: ✓ (1 flagged user out of 12,753)
- JSON reports saved: ✓

### Output Files Generated
| File | Size | Status |
|------|------|--------|
| `submission.csv` | 126.6 KB | ✓ Ready for submission |
| `submission_with_prob.csv` | 277.1 KB | ✓ Contains probabilities |
| `xai_reports.json` | 2.5 KB | ✓ Risk diagnosis report |

### Sample Output
- **Flagged User**: user_id 876703
- **Risk Probability**: 59.35% (MEDIUM tier)
- **Primary Risk Factors**:
  - Taiwan deposit count: 68.3% contribution
  - Late-night trading ratio: 30.4% contribution
- **Recommended Action**: Monitor for 60 days with transaction alerts

## Files Modified
1. `main_pipeline.py` - Enhanced XAI stage with better path resolution
2. `xai_bedrock.py` - Added NumpyEncoder for JSON serialization

## Verification
Run the pipeline with:
```bash
python main_pipeline.py --start-from xai --csv-dir ./data
```

Expected output:
- All three output files created successfully
- No errors or warnings
- Risk diagnosis report generated for flagged users
