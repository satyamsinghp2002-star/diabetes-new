import pickle
import numpy as np

scaler_path = 'Pickle/scaler.pkl'

try:
    scaler = pickle.load(open(scaler_path, 'rb'))
    print("✅ Scaler loaded successfully!")

    # Check feature count
    if hasattr(scaler, 'mean_'):
        print("Total features scaler trained on:", len(scaler.mean_))
        print("Feature means:", scaler.mean_)
    else:
        print("⚠️ Scaler object does not have 'mean_' attribute (maybe not fitted yet).")

except Exception as e:
    print("Error:", e)
