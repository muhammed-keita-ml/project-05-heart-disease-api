# app/main.py
# The FastAPI application — defines all API endpoints.

from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from app.model import load_model, load_feature_names, build_input_array
import pandas as pd
import numpy as np
from datetime import datetime
import json

# ── App initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Production ML API serving a Random Forest classifier "
        "trained on the UCI Heart Disease dataset. "
        "ROC-AUC: 0.921 | Disease Recall: 0.92"
    ),
    version="1.0.0",
)

# Load model and feature names at startup
# WHY at startup: Loading on every request would be very slow.
# Loading once when the app starts means predictions are fast.
model = load_model()
feature_names = load_feature_names()

# In-memory prediction log for drift monitoring
# WHY in-memory: Simple and sufficient for a portfolio project.
# In production this would be a database.
prediction_log = []

# Training data reference for drift detection
# We store mean values per feature from training as the reference distribution
TRAINING_REFERENCE = {
    "age": 53.5, "trestbps": 131.6, "chol": 199.1,
    "thalch": 137.5, "oldpeak": 0.88, "ca": 0.18
}

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Check if the API is running and the model is loaded.
    WHY: Load balancers and monitoring tools ping /health to verify
    the service is alive. Always include this in production APIs.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict heart disease risk from clinical measurements.
    Returns prediction (0/1), probability, and human-readable label.
    """
    try:
        # Convert request to dict and build input array
        data = request.model_dump()
        input_array = build_input_array(data, feature_names)

        # Run prediction
        prediction  = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])
        label = "Heart Disease Likely" if prediction == 1 else "No Heart Disease Detected"

        # Log prediction for monitoring
        # WHY: We need a history of predictions to detect drift over time
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "age":       data["age"],
            "trestbps":  data["trestbps"],
            "chol":      data["chol"],
            "thalch":    data["thalch"],
            "oldpeak":   data["oldpeak"],
            "ca":        data["ca"],
            "prediction": prediction,
            "probability": probability,
        }
        prediction_log.append(log_entry)

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            label=label,
            model="Random Forest (ROC-AUC=0.921)"
        )

    except Exception as e:
        # WHY 500 not 200: Never swallow errors silently.
        # Return a proper HTTP error so the caller knows something went wrong.
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats():
    """
    Return basic statistics on recent predictions.
    Shows prediction distribution and average input values.
    WHY: This is the simplest form of model monitoring —
    watching how inputs and outputs change over time.
    """
    if not prediction_log:
        return {"message": "No predictions logged yet.", "total_predictions": 0}

    df = pd.DataFrame(prediction_log)
    total = len(df)
    disease_rate = float(df["prediction"].mean())

    stats = {
        "total_predictions": total,
        "disease_rate": round(disease_rate, 4),
        "avg_age":      round(float(df["age"].mean()), 2),
        "avg_thalch":   round(float(df["thalch"].mean()), 2),
        "avg_oldpeak":  round(float(df["oldpeak"].mean()), 2),
        "avg_chol":     round(float(df["chol"].mean()), 2),
    }
    return stats


@app.get("/monitoring/drift", tags=["Monitoring"])
def monitoring_drift():
    """
    Detect data drift by comparing recent predictions
    against training data reference values.
    WHY: If incoming data looks very different from training data,
    model predictions may be unreliable. This endpoint flags that.
    """
    if len(prediction_log) < 5:
        return {
            "message": "Need at least 5 predictions to compute drift.",
            "total_predictions": len(prediction_log)
        }

    df = pd.DataFrame(prediction_log)
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

    drift_report = {}
    for col in numeric_cols:
        recent_mean = float(df[col].mean())
        reference_mean = TRAINING_REFERENCE.get(col, 0)
        pct_change = abs(recent_mean - reference_mean) / (reference_mean + 1e-9) * 100
        drift_report[col] = {
            "recent_mean":    round(recent_mean, 2),
            "reference_mean": reference_mean,
            "pct_change":     round(pct_change, 2),
            "drift_detected": pct_change > 20
        }

    any_drift = any(v["drift_detected"] for v in drift_report.values())

    return {
        "drift_detected": any_drift,
        "total_predictions": len(prediction_log),
        "feature_drift": drift_report
    }


@app.get("/", tags=["System"])
def root():
    """API root — returns basic info and available endpoints."""
    return {
        "name":        "Heart Disease Prediction API",
        "version":     "1.0.0",
        "model":       "Random Forest | ROC-AUC: 0.921 | Recall: 0.92",
        "docs":        "/docs",
        "health":      "/health",
        "predict":     "/predict",
        "monitoring":  "/monitoring/stats",
        "drift":       "/monitoring/drift",
    }