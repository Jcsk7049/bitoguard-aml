#!/usr/bin/env python3
import os
import sys
import xgboost as xgb

model_path = "model.json"
print(f"Current directory: {os.getcwd()}")
print(f"Model path: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        print("✓ Model loaded successfully")
        print(f"Model type: {type(model)}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("✗ Model file not found")
    sys.exit(1)
