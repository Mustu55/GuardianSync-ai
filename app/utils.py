"""Feature extraction and utility functions for SCADA anomaly detection."""

import numpy as np

FEATURE_COLUMNS = [
    "temperature", "pressure", "flow_rate", "voltage", "current",
    "rpm", "vibration", "humidity", "power_consumption", "response_time",
    "packet_size", "command_frequency", "error_rate", "network_latency"
]

# Normal operating ranges for SCADA sensors
NORMAL_RANGES = {
    "temperature":       (20.0, 85.0),
    "pressure":          (50.0, 200.0),
    "flow_rate":         (30.0, 150.0),
    "voltage":           (210.0, 250.0),
    "current":           (5.0, 30.0),
    "rpm":               (800, 3600),
    "vibration":         (0.5, 5.0),
    "humidity":          (20.0, 70.0),
    "power_consumption": (5.0, 50.0),
    "response_time":     (10, 100),
    "packet_size":       (64, 1024),
    "command_frequency": (1, 20),
    "error_rate":        (0.0, 0.05),
    "network_latency":   (5, 50),
}

# Hard limits for critical sensor values (force BLOCK if exceeded)
HARD_LIMITS = {
    "pressure": 200.0,
}


def extract_features(params: dict) -> np.ndarray:
    """Extract a feature vector from raw sensor parameters."""
    features = []
    for col in FEATURE_COLUMNS:
        features.append(float(params.get(col, 0.0)))
    return np.array(features).reshape(1, -1)


def evaluate_hard_limits(params: dict) -> list:
    """Return list of hard-limit violations."""
    violations = []
    for feature, max_value in HARD_LIMITS.items():
        value = float(params.get(feature, 0.0))
        if value > max_value:
            violations.append({
                "feature": feature,
                "value": value,
                "limit": max_value,
            })
    return violations


def compute_deviation_scores(params: dict) -> dict:
    """Compute per-feature deviation from normal range (0 = normal, >1 = anomalous)."""
    scores = {}
    for col in FEATURE_COLUMNS:
        val = float(params.get(col, 0.0))
        lo, hi = NORMAL_RANGES[col]
        mid = (lo + hi) / 2
        span = (hi - lo) / 2
        if span == 0:
            scores[col] = 0.0
        else:
            scores[col] = round(abs(val - mid) / span, 3)
    return scores


def get_top_anomalous_features(params: dict, top_n: int = 3) -> list:
    """Return the top-N features contributing most to anomaly."""
    scores = compute_deviation_scores(params)
    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[:top_n]
