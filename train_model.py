import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

print("Running file path:", __file__)

# 📁 Base directory (important)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 📁 Ensure Pickle folder exists
pickle_dir = os.path.join(base_dir, "Pickle")
os.makedirs(pickle_dir, exist_ok=True)

# 📥 Load dataset
data_path = os.path.join(base_dir, "diabetes.csv")
df = pd.read_csv(data_path)

# ❌ Remove Pregnancies
X = df.drop(["Outcome", "Pregnancies"], axis=1)
y = df["Outcome"]

# ⚙️ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔀 Split data
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🤖 Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 💾 Save paths
model_path = os.path.join(pickle_dir, "model_new.pkl")
scaler_path = os.path.join(pickle_dir, "scaler_new.pkl")

# 💾 Save files
pickle.dump(model, open(model_path, "wb"))
pickle.dump(scaler, open(scaler_path, "wb"))

# ✅ Done
print("Training Complete ✅")
print("Model saved at:", model_path)
print("Scaler saved at:", scaler_path)