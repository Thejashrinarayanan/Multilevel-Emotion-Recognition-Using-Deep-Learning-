import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# === Load dataset ===
data_path = "data/eeg_emotions_intensity.csv"
data = pd.read_csv(data_path)
print("✅ Data loaded successfully!")

# === Separate features and labels ===
X = data.drop(columns=["Emotion", "Intensity"])
y_emotion = data["Emotion"]
y_intensity = data["Intensity"]

# === Encode categorical labels ===
emotion_encoder = LabelEncoder()
intensity_encoder = LabelEncoder()

y_emotion_encoded = emotion_encoder.fit_transform(y_emotion)
y_intensity_encoded = intensity_encoder.fit_transform(y_intensity)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Save preprocessing objects for later use ===
os.makedirs("models", exist_ok=True)

with open("models/preprocessing_objects.pkl", "wb") as f:
    pickle.dump((scaler, emotion_encoder, intensity_encoder), f)

print("💾 Saved preprocessing_objects.pkl successfully!")
print(f"Features used for training: {list(X.columns)}")
