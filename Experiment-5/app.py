from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request


app = Flask(__name__)


BASE_DIR = Path(__file__).resolve().parent


MODEL_PATHS = [
    BASE_DIR.parent / "Experiment-2" / "occupancy_model.pkl",
    BASE_DIR / "occupancy_model.pkl"
]


FEATURES = [
    "S1_Temp",
    "S2_Temp",
    "S3_Temp",
    "S4_Temp",
    "S1_Light",
    "S2_Light",
    "S3_Light",
    "S4_Light",
    "S1_Sound",
    "S2_Sound",
    "S3_Sound",
    "S4_Sound",
    "S5_CO2",
    "S5_CO2_Slope",
    "S6_PIR",
    "S7_PIR"
]


def load_model():

    for model_path in MODEL_PATHS:

        if model_path.exists():

            return joblib.load(model_path)

    raise FileNotFoundError(
        "occupancy_model.pkl was not found."
    )


@app.get("/")
def root():

    return jsonify({
        "status": "ok",
        "service": "room-occupancy-prediction"
    })


@app.get("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "room-occupancy-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "error": "JSON request body is required"
        }), 400


    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]


    if missing_fields:

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400


    try:

        sample = pd.DataFrame([
            {
                feature: float(data[feature])
                for feature in FEATURES
            }
        ])

    except (TypeError, ValueError):

        return jsonify({
            "error": "All feature values must be numeric"
        }), 400


    model = load_model()

    prediction_code = int(
        model.predict(sample)[0]
    )


    return jsonify({
        "prediction": prediction_code,
        "prediction_code": prediction_code
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
