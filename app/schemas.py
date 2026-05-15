# app/schemas.py
# Pydantic models define the shape of API requests and responses.
# WHY Pydantic: FastAPI uses these to automatically validate incoming
# data, reject bad requests, and generate API documentation at /docs.

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Input schema for the /predict endpoint."""
    age:      float = Field(..., ge=1,   le=120, description="Age in years")
    trestbps: float = Field(..., ge=80,  le=250, description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)")
    thalch:   float = Field(..., ge=60,  le=220, description="Max heart rate achieved (bpm)")
    oldpeak:  float = Field(..., ge=0,   le=10,  description="ST depression induced by exercise")
    ca:       float = Field(..., ge=0,   le=3,   description="Number of major vessels (0-3)")
    sex:      str   = Field(..., description="Male or Female")
    cp:       str   = Field(..., description="Chest pain type")
    fbs:      str   = Field(..., description="Fasting blood sugar > 120 mg/dl: True or False")
    restecg:  str   = Field(..., description="Resting ECG results")
    exang:    str   = Field(..., description="Exercise induced angina: True or False")
    slope:    str   = Field(..., description="Slope of peak exercise ST segment")
    thal:     str   = Field(..., description="Thalassemia type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "trestbps": 145, "chol": 233,
                "thalch": 150, "oldpeak": 2.3, "ca": 0,
                "sex": "Male", "cp": "Typical angina",
                "fbs": "False", "restecg": "LV hypertrophy",
                "exang": "False", "slope": "Downsloping",
                "thal": "Fixed defect"
            }
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""
    prediction:  int   = Field(..., description="0 = No disease, 1 = Disease")
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    label:       str   = Field(..., description="Human readable result")
    model:       str   = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""
    status:       str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version:      str = Field(..., description="API version")