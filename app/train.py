"""Training pipeline: generate synthetic data, fit IsolationForest, save artifacts."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .utils import FEATURE_COLUMNS, extract_features
from .simulator import generate_normal_batch
from .attack_generator import generate_attack_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def generate_training_data(n_normal: int = 2400, n_attack: int = 500) -> pd.DataFrame:
    """Generate synthetic SCADA data for training."""
    industries = ["power_plant", "water_treatment", "manufacturing", "chemical_facility"]
    normal_rows, attack_rows = [], []

    for ind in industries:
        normals = generate_normal_batch(ind, n_normal // len(industries))
        for r in normals:
            row = r["parameters"].copy()
            row["label"] = 0
            row["industry"] = ind
            normal_rows.append(row)

        attacks = generate_attack_batch(ind, n_attack // len(industries))
        for r in attacks:
            row = r["parameters"].copy()
            row["label"] = 1
            row["industry"] = ind
            attack_rows.append(row)

    df = pd.DataFrame(normal_rows + attack_rows)
    return df


def train_model() -> dict:
    """Train the IsolationForest anomaly detector and save all artifacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_training_data()

    # Save datasets
    normal_df = df[df["label"] == 0]
    attack_df = df[df["label"] == 1]
    df.to_csv(os.path.join(DATA_DIR, "training_dataset.csv"), index=False)
    normal_df.to_csv(os.path.join(DATA_DIR, "normal_stream.csv"), index=False)
    attack_df.to_csv(os.path.join(DATA_DIR, "attack_stream.csv"), index=False)

    # Extract features (train only on normal data)
    X_normal = normal_df[FEATURE_COLUMNS].values
    X_all = df[FEATURE_COLUMNS].values

    # Fit scaler on normal data
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    # Train IsolationForest (contamination ~ expected attack ratio)
    model = IsolationForest(
        n_estimators=250,
        contamination=0.12,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_normal_scaled)

    # Evaluate on full dataset
    X_all_scaled = scaler.transform(X_all)
    preds = model.predict(X_all_scaled)
    labels = df["label"].values
    tp = sum((p == -1) and (l == 1) for p, l in zip(preds, labels))
    fp = sum((p == -1) and (l == 0) for p, l in zip(preds, labels))
    fn = sum((p == 1) and (l == 1) for p, l in zip(preds, labels))

    # Train a small neural classifier for supervised signal
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
    )
    mlp.fit(X_all_scaled, labels)

    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "anomaly_detector.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    columns_path = os.path.join(MODELS_DIR, "feature_columns.json")
    mlp_path = os.path.join(MODELS_DIR, "mlp_classifier.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(columns_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    joblib.dump(mlp, mlp_path)

    return {
        "status": "success",
        "message": f"Model trained on {len(X_normal)} normal samples",
        "samples_used": len(df),
        "model_path": model_path,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))
