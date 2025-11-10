import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model("models/eeg_emotion_intensity_model.h5")

# Load preprocessing objects
with open("models/preprocessing_objects.pkl", "rb") as f:
    scaler, emotion_encoder, intensity_encoder = pickle.load(f)

# Example input (replace with your actual EEG values)
input_data = np.array([[70, 65, 2, 60]])  # Attention, Meditation, EyeBlink, Average
input_scaled = scaler.transform(input_data)

# Prediction
emotion_pred, intensity_pred = model.predict(input_scaled)

emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
intensity_label = intensity_encoder.inverse_transform([np.argmax(intensity_pred)])[0]

print(f"🧠 Predicted Emotion: {emotion_label} 🔥")
print(f"Predicted Intensity: {intensity_label}")
