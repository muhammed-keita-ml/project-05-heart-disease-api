# tests/test_api.py
import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os

# ── Create mock model and feature names before importing app ──────────────────
# WHY: The real model file is not stored in the repo (too large).
# We mock it so CI/CD can run tests without the binary file.
# This is standard practice in production ML testing.

MOCK_FEATURE_NAMES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca",
    "sex_Male",
    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",
    "fbs_True",
    "restecg_normal",
    "restecg_st-t abnormality",
    "exang_True",
    "slope_flat",
    "slope_upsloping",
    "thal_normal",
    "thal_reversable defect",
]

# Create mock model files in app/models/ for testing
os.makedirs("app/models", exist_ok=True)
with open("app/models/feature_names.json", "w") as f:
    json.dump(MOCK_FEATURE_NAMES, f)

# Create a mock sklearn model and save it
import joblib
from sklearn.ensemble import RandomForestClassifier

mock_model = RandomForestClassifier(n_estimators=2, random_state=42)
X_dummy = np.zeros((10, len(MOCK_FEATURE_NAMES)))
y_dummy = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
mock_model.fit(X_dummy, y_dummy)
joblib.dump(mock_model, "app/models/heart_disease_model.pkl")

# Now import the app after mock files exist
from app.main import app

client = TestClient(app)

# ── Sample valid request payload ──────────────────────────────────────────────
VALID_PAYLOAD = {
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalch": 150,
    "oldpeak": 2.3,
    "ca": 0,
    "sex": "Male",
    "cp": "Typical angina",
    "fbs": "False",
    "restecg": "LV hypertrophy",
    "exang": "False",
    "slope": "Downsloping",
    "thal": "Fixed defect",
}


def test_root():
    """Root endpoint returns 200 and expected keys."""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "model" in response.json()


def test_health():
    """Health endpoint returns healthy status and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_predict_valid():
    """Valid payload returns prediction with correct structure."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_no_disease():
    """Healthy patient profile returns valid prediction."""
    healthy_payload = {
        "age": 35,
        "trestbps": 110,
        "chol": 180,
        "thalch": 185,
        "oldpeak": 0.0,
        "ca": 0,
        "sex": "Female",
        "cp": "Non-anginal",
        "fbs": "False",
        "restecg": "Normal",
        "exang": "False",
        "slope": "Upsloping",
        "thal": "Normal",
    }
    response = client.post("/predict", json=healthy_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]


def test_predict_missing_field():
    """Request missing required field returns 422 validation error."""
    bad_payload = {"age": 63, "trestbps": 145}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_monitoring_stats():
    """Monitoring stats returns total_predictions field."""
    response = client.get("/monitoring/stats")
    assert response.status_code == 200
    assert "total_predictions" in response.json()


def test_monitoring_drift():
    """Drift endpoint returns valid response."""
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data or "drift_detected" in data
