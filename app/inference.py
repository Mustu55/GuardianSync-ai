"""Inference pipeline: load model and predict anomalies."""

import os
import joblib
import json
import numpy as np
import logging
from .utils import FEATURE_COLUMNS, extract_features, evaluate_hard_limits

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_model = None
_scaler = None
_mlp = None


def load_model():
    """Load the trained model and scaler from disk."""
    global _model, _scaler, _mlp
    model_path = os.path.join(MODELS_DIR, "anomaly_detector.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    mlp_path = os.path.join(MODELS_DIR, "mlp_classifier.pkl")

    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _model = joblib.load(model_path)
            _scaler = joblib.load(scaler_path)
            if os.path.exists(mlp_path):
                _mlp = joblib.load(mlp_path)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.warning(f"Model files not found at {MODELS_DIR}")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def is_model_loaded() -> bool:
    return _model is not None and _scaler is not None


def predict(params: dict) -> dict:
    """Run anomaly detection on sensor parameters.
    
    Returns dict with: is_anomaly, anomaly_score (0–1), raw_score
    """
    try:
        if not params or not isinstance(params, dict):
            raise ValueError("Invalid parameters")

        # Hard safety limits remain enforceable while the optional trained model is unavailable.
        hard_limits = evaluate_hard_limits(params)
        if hard_limits:
            logger.warning(f"Hard limit violations: {hard_limits}")
            return {
                "is_anomaly": True,
                "anomaly_score": 1.0,
                "raw_score": 1.0,
                "hard_limits": hard_limits,
            }
        
        if not is_model_loaded():
            loaded = load_model()
            if not loaded:
                return {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "raw_score": 0.0,
                    "error": "Model not trained yet. Call /train first."
                }

        # Extract and validate features
        features = extract_features(params)
        if features is None or len(features[0]) != len(FEATURE_COLUMNS):
            raise ValueError("Feature extraction failed")
        
        features_scaled = _scaler.transform(features)

        # IsolationForest: decision_function returns negative for anomalies
        raw_score = _model.decision_function(features_scaled)[0]

        # Convert raw_score to 0–1 anomaly score (higher = more anomalous)
        # Use a sigmoid to avoid overly aggressive blocking for minor deviations
        anomaly_score = 1.0 / (1.0 + np.exp(raw_score * 4.0))
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # Ensemble with MLP if available
        if _mlp is not None:
            try:
                proba = _mlp.predict_proba(features_scaled)[0][1]
                anomaly_score = 0.6 * anomaly_score + 0.4 * float(proba)
            except Exception as e:
                logger.warning(f"MLP prediction failed: {str(e)}")

        is_anomaly = anomaly_score >= 0.75

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "raw_score": round(float(raw_score), 4),
            "hard_limits": [],
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "raw_score": 0.0,
            "error": f"Prediction failed: {str(e)}"
        }
