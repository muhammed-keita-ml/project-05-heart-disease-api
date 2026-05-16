# Project 05 — Heart Disease Prediction API

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue)
![CI/CD](https://github.com/muhammed-keita-ml/project-05-heart-disease-api/actions/workflows/ci.yml/badge.svg)
![Railway](https://img.shields.io/badge/Railway-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A production-grade ML monitoring API serving the heart disease Random Forest
classifier from Project 04. Built with FastAPI, containerised with Docker,
tested with pytest, deployed via GitHub Actions CI/CD to Railway.

**🚀 Live API:** https://project-05-heart-disease-api-production.up.railway.app  
**📖 Interactive Docs:** https://project-05-heart-disease-api-production.up.railway.app/docs  
**💻 Project 04 (model source):** https://github.com/muhammed-keita-ml/project-04-heart-disease-pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CI/CD Layer                          │
│  GitHub Actions → Run Tests → Build Docker → Deploy        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Application Layer                         │
│                                                             │
│   FastAPI Application (uvicorn)                             │
│   ├── GET  /           → API info                           │
│   ├── GET  /health     → Health check                       │
│   ├── POST /predict    → Heart disease prediction           │
│   ├── GET  /monitoring/stats  → Prediction statistics       │
│   └── GET  /monitoring/drift  → Data drift detection        │
│                                                             │
│   Model Loader                                              │
│   ├── RandomForestClassifier (trained at build time)        │
│   └── feature_names.json (column order guarantee)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Monitoring Layer                          │
│   Prediction Logger  → logs every request                   │
│   Drift Detector     → compares recent vs training dist.    │
│   Alert System       → flags features with >20% drift       │
└─────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

```
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check — model loaded status |
| POST | `/predict` | Predict heart disease from clinical inputs |
| GET | `/monitoring/stats` | Prediction statistics and averages |
| GET | `/monitoring/drift` | Data drift detection report |

```

---

## Quick Start

### Test the live API with curl

**Health check:**
```bash
curl https://project-05-heart-disease-api-production.up.railway.app/health
```

**Make a prediction:**
```bash
curl -X POST \
  https://project-05-heart-disease-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": "Fixed defect"
  }'
```

**Expected response:**
```json
{
  "prediction": 1,
  "probability": 0.72,
  "label": "Heart Disease Likely",
  "model": "Random Forest (ROC-AUC=0.921)"
}
```

**Check monitoring stats:**
```bash
curl https://project-05-heart-disease-api-production.up.railway.app/monitoring/stats
```

**Check drift detection:**
```bash
curl https://project-05-heart-disease-api-production.up.railway.app/monitoring/drift
```

---

## Input Schema

```
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| age | float | 1–120 | Age in years |
| trestbps | float | 80–250 | Resting blood pressure (mmHg) |
| chol | float | 100–700 | Serum cholesterol (mg/dl) |
| thalch | float | 60–220 | Max heart rate achieved (bpm) |
| oldpeak | float | 0–10 | ST depression induced by exercise |
| ca | float | 0–3 | Number of major vessels coloured |
| sex | string | Male/Female | Biological sex |
| cp | string | Asymptomatic/Atypical angina/Non-anginal/Typical angina | Chest pain type |
| fbs | string | True/False | Fasting blood sugar > 120 mg/dl |
| restecg | string | Normal/ST-T abnormality/LV hypertrophy | Resting ECG results |
| exang | string | True/False | Exercise induced angina |
| slope | string | Flat/Upsloping/Downsloping | ST segment slope |
| thal | string | Normal/Fixed defect/Reversable defect | Thalassemia type |

```

---

## Monitoring System

Every prediction request is logged with inputs, output, probability, and
timestamp. Two monitoring endpoints provide visibility into model behaviour
in production:

**`/monitoring/stats`** — returns total predictions, disease rate, and
average values for key continuous features. Use this to track whether the
model is seeing unusual patient profiles.

**`/monitoring/drift`** — compares recent prediction inputs against training
data reference values. Flags any feature where the recent mean has shifted
more than 20% from the training distribution.

```json
{
  "drift_detected": false,
  "total_predictions": 42,
  "feature_drift": {
    "age": {
      "recent_mean": 54.2,
      "reference_mean": 53.5,
      "pct_change": 1.31,
      "drift_detected": false
    }
  }
}
```

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/muhammed-keita-ml/project-05-heart-disease-api
cd project-05-heart-disease-api

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Start the API
uvicorn app.main:app --reload

# Visit http://127.0.0.1:8000/docs
```

---

## Run with Docker

```bash
# Build the image (trains model automatically)
docker build -t heart-disease-api .

# Run the container
docker run -p 8000:8000 heart-disease-api

# Visit http://localhost:8000/docs
```

---

## Run Tests

```bash
pytest tests/ -v
```

All 7 tests cover: root endpoint, health check, valid prediction,
prediction structure validation, missing field rejection (422),
monitoring stats, and drift detection.

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions pipeline:

```
Push to main
    │
    ▼
Install dependencies
    │
    ▼
Run pytest (7 tests)
    │
    ▼
Build Docker image
    │
    ▼
✅ Pipeline passes → Railway auto-deploys
```

---

## Project Structure

```
project-05-heart-disease-api/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application + endpoints
│   ├── model.py                # Model loading + input array builder
│   ├── schemas.py              # Pydantic request/response schemas
│   └── models/
│       ├── heart_disease_model.pkl   # Trained RF model
│       └── feature_names.json        # Feature column order
├── tests/
│   ├── __init__.py
│   └── test_api.py             # pytest test suite (7 tests)
├── .gitignore
├── Dockerfile                  # Container definition
├── README.md
├── requirements.txt
└── train_model.py              # Model training script
```

---

## Tools & Technologies

| Category | Tool |
|---|---|
| API Framework | FastAPI + uvicorn |
| ML Model | scikit-learn Random Forest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Deployment | Railway |
| Testing | pytest + httpx |
| Validation | Pydantic v2 |
| Model serialisation | joblib |
| License | MIT |

---

## Model Performance

The underlying model is trained on the UCI Heart Disease dataset:

| Metric | Score |
|---|---|
| ROC-AUC | 0.921 |
| Disease Recall | 0.92 |
| Accuracy | 86% |
| Training data | 920 patients, 4 hospitals |

Full training details in [Project 04](https://github.com/muhammed-keita-ml/project-04-heart-disease-pipeline).

---

## License

This project is licensed under the MIT License.

---

## Author

**Muhammed Keita** — ML Engineer in Training  
[GitHub](https://github.com/muhammed-keita-ml) ·
[LinkedIn](https://linkedin.com/in/muhammed-keita) ·
[Hugging Face](https://huggingface.co/muhammed-keita-ml)

---

*Part of an end-to-end ML/MLOps portfolio. Building in public from The Gambia 🇬🇲*