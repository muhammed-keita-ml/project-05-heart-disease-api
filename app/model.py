# app/model.py
# Responsible for loading the model and feature names from disk.
# Isolated here so main.py stays clean and this logic is testable independently.

import joblib
import json
import numpy as np
from pathlib import Path

# Build path relative to this file so it works inside Docker too
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_model():
    """Load the trained Random Forest model from disk."""
    model_path = MODELS_DIR / "heart_disease_model.pkl"
    model = joblib.load(model_path)
    return model


def load_feature_names():
    """Load the feature column order used during training."""
    feature_path = MODELS_DIR / "feature_names.json"
    with open(feature_path, "r") as f:
        feature_names = json.load(f)
    return feature_names


def build_input_array(data: dict, feature_names: list) -> np.ndarray:
    """
    Convert a raw input dict into a numpy array in the exact
    column order the model was trained on.

    WHY: The model expects features in a specific order. If any column
    is in the wrong position, predictions are silently wrong.
    This function guarantees correct ordering every time.
    """
    input_dict = {col: 0 for col in feature_names}

    # Continuous features — fill directly
    input_dict["age"]      = data.get("age", 0)
    input_dict["trestbps"] = data.get("trestbps", 0)
    input_dict["chol"]     = data.get("chol", 0)
    input_dict["thalch"]   = data.get("thalch", 0)
    input_dict["oldpeak"]  = data.get("oldpeak", 0)
    input_dict["ca"]       = data.get("ca", 0)

    # Categorical features — set matching one-hot column to 1
    if data.get("sex") == "Male":
        input_dict["sex_Male"] = 1

    cp_map = {
        "Atypical angina": "cp_atypical angina",
        "Non-anginal":     "cp_non-anginal",
        "Typical angina":  "cp_typical angina",
    }
    cp = data.get("cp", "")
    if cp in cp_map and cp_map[cp] in input_dict:
        input_dict[cp_map[cp]] = 1

    if data.get("fbs") == "True":
        input_dict["fbs_True"] = 1

    restecg_map = {
        "Normal":           "restecg_normal",
        "ST-T abnormality": "restecg_st-t abnormality",
    }
    restecg = data.get("restecg", "")
    if restecg in restecg_map and restecg_map[restecg] in input_dict:
        input_dict[restecg_map[restecg]] = 1

    if data.get("exang") == "True":
        input_dict["exang_True"] = 1

    slope_map = {
        "Flat":      "slope_flat",
        "Upsloping": "slope_upsloping",
    }
    slope = data.get("slope", "")
    if slope in slope_map and slope_map[slope] in input_dict:
        input_dict[slope_map[slope]] = 1

    thal_map = {
        "Normal":            "thal_normal",
        "Reversable defect": "thal_reversable defect",
    }
    thal = data.get("thal", "")
    if thal in thal_map and thal_map[thal] in input_dict:
        input_dict[thal_map[thal]] = 1

    # Build array in correct column order
    input_array = np.array([[input_dict[col] for col in feature_names]])
    return input_array