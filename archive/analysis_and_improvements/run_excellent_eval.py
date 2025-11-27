import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(
        f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f} | "
        f"Macro F1: {report['macro avg']['f1-score']:.4f} | "
        f"Improving F1: {report['2']['f1-score']:.4f}"
    )

    return {
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": roc,
        "Macro_F1": report["macro avg"]["f1-score"],
        "Improving_F1": report["2"]["f1-score"],
    }


def main():
    print("Loading enhanced dataset...")
    df = pd.read_csv("../today/trajectory_ml_ready_excellent.csv")

    X = df.drop(
        columns=[
            "UNITID",
            "Institution_Name",
            "Year",
            "State",
            "Target_Trajectory",
            "Target_Label",
        ]
    )
    y = df["Target_Label"].astype(int)

    categorical_cols = ["Division", "Lag1_Division"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    results = []

    log_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, multi_class="multinomial"),
            ),
        ]
    )
    log_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(log_pipeline, X_test, y_test, "Logistic Regression (SMOTE)"))

    rf_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(rf_pipeline, X_test, y_test, "Random Forest (SMOTE)"))

    xgb_params = {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    xgb_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )
    xgb_pipeline.fit(X_train, y_train)
    results.append(evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost (Enhanced)"))

    results_df = pd.DataFrame(results)
    print("\nFinal Comparison:")
    print(results_df)

    best = results_df.sort_values("Accuracy", ascending=False).iloc[0]
    best_model = {
        "Logistic Regression (SMOTE)": log_pipeline,
        "Random Forest (SMOTE)": rf_pipeline,
        "XGBoost (Enhanced)": xgb_pipeline,
    }[best["Model"]]

    out_path = "../today/models/final_trajectory_model_excellent.joblib"
    joblib.dump(best_model, out_path)
    print(f"\nSaved best model ({best['Model']}) to {out_path}")


if __name__ == "__main__":
    main()

