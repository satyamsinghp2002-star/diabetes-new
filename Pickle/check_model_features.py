import pickle

# Purana model load karo
model_path = 'Pickle/model.pkl'
model = pickle.load(open(model_path, 'rb'))

try:
    # Agar model ke paas feature names hain
    if hasattr(model, 'feature_names_in_'):
        print("✅ Model trained with features:")
        print(model.feature_names_in_)
        print("Total features:", len(model.feature_names_in_))
    else:
        print("⚠️ Model ke paas feature_names_in_ attribute nahi hai.")
        print("Ho sakta hai ye purane sklearn version se train hua ho.")
except Exception as e:
    print("Error:", e)
