#!/usr/bin/env python3
"""
NCAA Financial Trajectory Prediction CLI

Usage:
    python predict.py "School Name"
    python predict.py 123456  # UNITID

Note: Run with the project's virtual environment Python:
    .venv/bin/python predict.py "Alabama"
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent
MODEL_PATH = ASSETS_DIR / "models" / "trajectory_model.joblib"
DATA_PATH = ASSETS_DIR / "data" / "trajectory_excellent.csv"


def load_model():
    """Load the trained XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_data():
    """Load the prediction dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def predict_school(query):
    """Predict trajectory for a school by name or UNITID."""
    model = load_model()
    df = load_data()

    # Search by UNITID or name
    if str(query).isdigit():
        matches = df[df["UNITID"] == int(query)]
    else:
        matches = df[df["Institution_Name"].str.contains(query, case=False, na=False)]

    if matches.empty:
        print(f"No school found matching '{query}'")
        return None

    # Use latest year
    latest = matches.loc[matches["Year"].idxmax()]
    print(f"\n--- Prediction for {latest['Institution_Name']} ({int(latest['Year'])}) ---")

    # Prepare features
    drop_cols = ["UNITID", "Institution_Name", "Year", "Target_Trajectory", "Target_Label", "State"]
    X = pd.DataFrame([latest]).drop(columns=[c for c in drop_cols if c in latest.index])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    labels = {0: "Declining", 1: "Stable", 2: "Improving"}

    print(f"Predicted Trajectory: {labels[pred]}")
    print(f"Confidence: Declining {proba[0]:.2%} | Stable {proba[1]:.2%} | Improving {proba[2]:.2%}")

    return labels[pred]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <School Name or UNITID>")
        print('Example: python predict.py "Alabama"')
        sys.exit(1)

    predict_school(sys.argv[1])
