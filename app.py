from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
import subprocess

app = Flask(__name__)

# === Load model and preprocessing objects ===
model_path = os.path.join("models", "eeg_emotion_intensity_model.h5")
model = load_model(model_path)

# Load scaler and encoders
with open("models/preprocessing_objects.pkl", "rb") as f:
    scaler, emotion_encoder, intensity_encoder = pickle.load(f)

# === Home page ===
@app.route('/')
def home():
    # Render page initially with no predictions
    return render_template('index.html', emotion=None, intensity=None, error=None)


def run_script(script_path):
    """
    Run a Python script and return its stdout and stderr as a string.
    """
    result = subprocess.run(
        ["python", script_path],  # run the script
        capture_output=True,      # capture output instead of printing to terminal
        text=True                 # return output as string, not bytes
    )
    
    # Combine stdout and stderr
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return output



# === Prediction route ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        attention = float(request.form['attention'])
        meditation = float(request.form['meditation'])
        eyeblink = float(request.form['eyeblink'])
        average = float(request.form['average'])

        # Prepare and scale input
        input_data = np.array([[attention, meditation, eyeblink, average]])
        input_scaled = scaler.transform(input_data)

        # Multi-output model prediction
        emotion_pred, intensity_pred = model.predict(input_scaled)

        # Decode predictions
        emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
        intensity_label = intensity_encoder.inverse_transform([np.argmax(intensity_pred)])[0]

        # Render results on the same page
        return render_template(
            "index.html",
            emotion=emotion_label,
            intensity=intensity_label,
            error=None
        )

    except Exception as e:
        return render_template("index.html", emotion=None, intensity=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
