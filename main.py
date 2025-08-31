import json
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# Load saved artifacts
model = tf.keras.models.load_model("heart_model.keras")
scaler = joblib.load("scaler.joblib")
with open("features.json", "r") as f:
    feature_names = json.load(f)

app = FastAPI(title="❤️ Heart Disease Predictor")


class PatientData(BaseModel):
    data: dict 

@app.post("/predict")
def predict(patient: PatientData):
    input_dict = patient.data
    row = [input_dict.get(col, 0) for col in feature_names] 
    
    X_scaled = scaler.transform([row])
    prob = model.predict(X_scaled)[0][0]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability": float(prob),
        "message": "Likely Heart Disease" if pred == 1 else "Unlikely Heart Disease"
    }
