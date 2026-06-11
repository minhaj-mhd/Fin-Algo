import xgboost as xgb
import json

model_path = 'models/v17_random_forest_1h/xgb_long_model.json'
meta_path = 'models/v17_random_forest_1h/metadata.json'

with open(meta_path, 'r') as f:
    meta = json.load(f)
features = meta.get('features', [])

bst = xgb.Booster()
bst.load_model(model_path)
bst.feature_names = features

dump = bst.get_dump()
if dump:
    print("--- FINAL DECISION TREE (LONG MODEL) ---")
    print(dump[-1])
else:
    print("No trees found.")
