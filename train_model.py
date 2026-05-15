# train_model.py
# Retrains the Random Forest model and saves it to app/models/
# Run this script once to generate a compatible model file.

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading data...")
url = "https://raw.githubusercontent.com/muhammed-keita-ml/project-04-heart-disease-pipeline/main/heart_disease_uci.csv"

try:
    df = pd.read_csv(url)
except:
    # Fallback — create synthetic data matching the feature structure
    print("Using synthetic training data...")
    np.random.seed(42)
    n = 920
    df = pd.DataFrame(
        {
            "age": np.random.randint(29, 77, n),
            "trestbps": np.random.randint(94, 200, n).astype(float),
            "chol": np.random.randint(126, 564, n).astype(float),
            "thalch": np.random.randint(71, 202, n).astype(float),
            "oldpeak": np.random.uniform(0, 6.2, n),
            "ca": np.random.randint(0, 4, n).astype(float),
            "sex": np.random.choice(["Male", "Female"], n),
            "cp": np.random.choice(
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"], n
            ),
            "fbs": np.random.choice([True, False], n),
            "restecg": np.random.choice(
                ["normal", "lv hypertrophy", "st-t abnormality"], n
            ),
            "exang": np.random.choice([True, False], n),
            "slope": np.random.choice(["flat", "upsloping", "downsloping"], n),
            "thal": np.random.choice(
                ["normal", "fixed defect", "reversable defect"], n
            ),
            "num": np.random.randint(0, 5, n),
        }
    )

# Clean and prepare
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

object_cols = df.select_dtypes(include="object").columns.tolist()
bool_cols = df.select_dtypes(include="bool").columns.tolist()

for col in bool_cols:
    df[col] = df[col].astype(str)
    object_cols.append(col)

for col in object_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, columns=object_cols, drop_first=True)

numeric_cols = [c for c in df.columns if c != "target"]
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

bool_dtype_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_dtype_cols] = df[bool_dtype_cols].astype(int)

X = df.drop(columns=["target"])
y = df["target"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values, y_train)

os.makedirs("app/models", exist_ok=True)
joblib.dump(model, "app/models/heart_disease_model.pkl")

with open("app/models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print(f"Model saved. Features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
