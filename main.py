import numpy as np
import pickle
from tensorflow.keras.models import load_model

# === Load trained model ===
model = load_model('models/eeg_emotion_intensity_model.h5')

# === Load preprocessing objects (scaler + encoders) ===
with open('preprocessing_objects.pkl', 'rb') as f:
    scaler, emotion_encoder, intensity_encoder = pickle.load(f)

print("✅ Model and preprocessing objects loaded successfully!\n")

# === Example EEG sample input ===
# IMPORTANT: Must match the number of features used during training.
# You can check this dynamically below.
print("Scaler expects features:", scaler.feature_names_in_)

# Create a sample with correct number of features (replace with your actual EEG values)
sample = np.array([[0.25, 0.48, 0.62, 0.33]])  # Example with 4 features

print("Your sample shape:", sample.shape)

# === Preprocess the sample ===
sample_scaled = scaler.transform(sample)

# === Make predictions ===
pred_emotion, pred_intensity = model.predict(sample_scaled)

# === Decode predictions ===
emotion_pred = np.argmax(pred_emotion, axis=1)
intensity_pred = np.argmax(pred_intensity, axis=1)

emotion_label = emotion_encoder.inverse_transform(emotion_pred)[0]
intensity_label = intensity_encoder.inverse_transform(intensity_pred)[0]

# === Display output ===
print("\n🎯 Prediction Results:")
print(f"Predicted Emotion: {emotion_label}")
print(f"Predicted Intensity: {intensity_label}")
