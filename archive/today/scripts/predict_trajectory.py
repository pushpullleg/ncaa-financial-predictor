import sys
from pathlib import Path

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_path(candidates):
    """Return the first existing path from a list of candidate filenames."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected files were found: "
        + ", ".join(str(path) for path in candidates)
    )


def load_model():
    model_candidates = [
        PROJECT_ROOT / "models" / "final_trajectory_model_excellent.joblib",
        PROJECT_ROOT / "final_trajectory_model_excellent.joblib",
        PROJECT_ROOT / "models" / "final_trajectory_model.joblib",
        PROJECT_ROOT / "final_trajectory_model.joblib",
    ]

    model_path = _resolve_path(model_candidates)
    return joblib.load(model_path)


def load_data():
    dataset_candidates = [
        PROJECT_ROOT / "trajectory_ml_ready_excellent.csv",
        PROJECT_ROOT / "trajectory_ml_ready_advanced.csv",
        PROJECT_ROOT / "trajectory_ml_ready.csv",
    ]

    data_path = _resolve_path(dataset_candidates)
    return pd.read_csv(data_path)

def predict_school(school_name_or_id):
    model = load_model()
    df = load_data()
    
    # Search for the school
    if str(school_name_or_id).isdigit():
        school_data = df[df['UNITID'] == int(school_name_or_id)]
    else:
        # Case insensitive search
        school_data = df[df['Institution_Name'].str.contains(school_name_or_id, case=False, na=False)]
    
    if school_data.empty:
        print(f"No school found matching '{school_name_or_id}'")
        return

    # Get the latest year for this school
    latest_year = school_data['Year'].max()
    latest_data = school_data[school_data['Year'] == latest_year].iloc[0]
    
    print(f"\n--- Prediction for {latest_data['Institution_Name']} ({latest_year}) ---")
    
    # Prepare features for prediction
    # We need to ensure the input DataFrame has the same columns as X used in training
    # The pipeline expects 'Division' and numerical columns.
    # We need to drop the identifiers.
    
    input_data = pd.DataFrame([latest_data])
    drop_cols = ['UNITID', 'Institution_Name', 'Year', 'Target_Trajectory', 'Target_Label', 'State']
    X_input = input_data.drop(columns=drop_cols)
    
    # Predict
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    
    # Map label to class name
    label_map = {0: 'Declining', 1: 'Stable', 2: 'Improving'}
    predicted_class = label_map[prediction]
    
    print(f"Predicted Trajectory: {predicted_class}")
    print(f"Confidence Scores:")
    print(f"  Declining: {probabilities[0]:.2%}")
    print(f"  Stable:    {probabilities[1]:.2%}")
    print(f"  Improving: {probabilities[2]:.2%}")
    
    print("\nKey Metrics (Latest Year):")
    print(f"  Total Revenue: ${latest_data['Grand Total Revenue']:,.0f}")
    print(f"  Total Expenses: ${latest_data['Grand Total Expenses']:,.0f}")
    print(f"  Efficiency Ratio: {latest_data['Efficiency_Mean_2yr']:.2f}")
    print(f"  Revenue Growth (1yr): {latest_data['Revenue_Growth_1yr']:.2%}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_trajectory.py <School Name or UNITID>")
        print("Example: python predict_trajectory.py \"Alabama\"")
    else:
        school_query = sys.argv[1]
        predict_school(school_query)
