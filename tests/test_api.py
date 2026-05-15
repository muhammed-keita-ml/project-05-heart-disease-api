# tests/test_api.py
# Automated tests for all API endpoints.
# WHY pytest: Tests run automatically in CI/CD — if any test fails,
# deployment stops. This prevents broken code reaching production.

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# ── Sample valid request payload ──────────────────────────────────────────────
VALID_PAYLOAD = {
    "age": 63, "trestbps": 145, "chol": 233,
    "thalch": 150, "oldpeak": 2.3, "ca": 0,
    "sex": "Male", "cp": "Typical angina",
    "fbs": "False", "restecg": "LV hypertrophy",
    "exang": "False", "slope": "Downsloping",
    "thal": "Fixed defect"
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
    """Healthy patient profile should return low disease probability."""
    healthy_payload = {
        "age": 35, "trestbps": 110, "chol": 180,
        "thalch": 185, "oldpeak": 0.0, "ca": 0,
        "sex": "Female", "cp": "Non-anginal",
        "fbs": "False", "restecg": "Normal",
        "exang": "False", "slope": "Upsloping",
        "thal": "Normal"
    }
    response = client.post("/predict", json=healthy_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probability"] < 0.6


def test_predict_missing_field():
    """Request missing required field returns 422 validation error."""
    bad_payload = {"age": 63, "trestbps": 145}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_monitoring_stats_empty():
    """Monitoring stats returns total_predictions field."""
    response = client.get("/monitoring/stats")
    assert response.status_code == 200
    assert "total_predictions" in response.json()


def test_monitoring_drift_insufficient():
    """Drift endpoint requires at least 5 predictions."""
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    assert "message" in response.json() or "drift_detected" in response.json()