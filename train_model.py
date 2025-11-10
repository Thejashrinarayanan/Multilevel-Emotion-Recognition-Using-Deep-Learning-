import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import os

# === Load dataset ===
data_path = "data/eeg_emotions_intensity.csv"
data = pd.read_csv(data_path)
print("Data loaded successfully!")

# === Features and labels ===
X = data[['Attention', 'Meditation', 'Eye Blink', 'Average']].values
y_emotion = data['Emotion'].values
y_intensity = data['Intensity'].values

# === Encode labels ===
emotion_encoder = LabelEncoder()
intensity_encoder = LabelEncoder()

y_emotion_encoded = emotion_encoder.fit_transform(y_emotion)
y_intensity_encoded = intensity_encoder.fit_transform(y_intensity)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/test split ===
X_train, X_test, y_emotion_train, y_emotion_test, y_intensity_train, y_intensity_test = train_test_split(
    X_scaled, y_emotion_encoded, y_intensity_encoded, test_size=0.2, random_state=42
)

# === Build simple multi-output model ===
inputs = Input(shape=(X_train.shape[1],))

# Shared dense layers
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# Emotion output
emotion_output = Dense(len(np.unique(y_emotion_encoded)), activation='softmax', name='emotion_output')(x)

# Intensity output
intensity_output = Dense(len(np.unique(y_intensity_encoded)), activation='softmax', name='intensity_output')(x)

model = Model(inputs=inputs, outputs=[emotion_output, intensity_output])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'emotion_output': 'sparse_categorical_crossentropy', 'intensity_output': 'sparse_categorical_crossentropy'},
    metrics={'emotion_output': 'accuracy', 'intensity_output': 'accuracy'}
)

model.summary()

# === Train model ===
history = model.fit(
    X_train,
    [y_emotion_train, y_intensity_train],
    validation_split=0.2,
    epochs=30,
    batch_size=32
)

# === Evaluate on test set ===
test_metrics = model.evaluate(X_test, [y_emotion_test, y_intensity_test])
print(f"Test metrics: {test_metrics}")

# === Save model and preprocessing objects ===
os.makedirs("models", exist_ok=True)
model.save("models/eeg_emotion_intensity_model.h5")
with open("models/preprocessing_objects.pkl", "wb") as f:
    pickle.dump((scaler, emotion_encoder, intensity_encoder), f)

print("Model and preprocessing objects saved successfully!")
