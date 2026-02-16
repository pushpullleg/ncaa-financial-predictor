"""Unit tests for NCAA Financial Trajectory prediction pipeline."""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "final" / "assets" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from predict import load_model, load_data, predict_school, MODEL_PATH, DATA_PATH


class TestDataLoading:
    """Tests for data and model loading functions."""

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model file not found at {MODEL_PATH}"

    def test_data_file_exists(self):
        assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"

    def test_load_model_returns_object(self):
        model = load_model()
        assert model is not None
        assert hasattr(model, "predict"), "Model must have a predict method"
        assert hasattr(model, "predict_proba"), "Model must have predict_proba method"

    def test_load_data_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Dataset should not be empty"

    def test_data_has_required_columns(self):
        df = load_data()
        required = ["UNITID", "Institution_Name", "Year"]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"


class TestPrediction:
    """Tests for the prediction logic."""

    def test_predict_known_school(self):
        result = predict_school("Alabama")
        assert result in ["Declining", "Stable", "Improving"]

    def test_predict_by_partial_name(self):
        result = predict_school("Texas")
        assert result in ["Declining", "Stable", "Improving"]

    def test_predict_unknown_school_returns_none(self):
        result = predict_school("NonexistentSchool12345")
        assert result is None

    def test_predict_by_unitid(self):
        df = load_data()
        unitid = str(int(df["UNITID"].iloc[0]))
        result = predict_school(unitid)
        assert result in ["Declining", "Stable", "Improving"]

    def test_prediction_labels_are_valid(self):
        """Ensure model only outputs valid class labels."""
        model = load_model()
        df = load_data()
        drop_cols = ["UNITID", "Institution_Name", "Year", "Target_Trajectory", "Target_Label", "State"]
        sample = df.iloc[:5]
        X = sample.drop(columns=[c for c in drop_cols if c in sample.columns])
        preds = model.predict(X)
        assert all(p in [0, 1, 2] for p in preds)
