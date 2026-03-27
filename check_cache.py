import pandas as pd

# Load feature cache
cache = pd.read_parquet('feature_cache.parquet')
print("Feature cache shape:", cache.shape)
print("\nStatus distribution:")
print(cache['status'].value_counts())
print("\nFirst few rows:")
print(cache.head())
print("\nTrain/Predict split:")
print(cache['_split'].value_counts())
print("\nTrain data status distribution:")
train_cache = cache[cache['_split'] == 'train']
print(train_cache['status'].value_counts())
