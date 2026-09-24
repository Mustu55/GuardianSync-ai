"""Explainability module: generate human-readable anomaly explanations."""

from .utils import get_top_anomalous_features, NORMAL_RANGES, FEATURE_COLUMNS, HARD_LIMITS

FEATURE_LABELS = {
    "temperature": "Temperature",
    "pressure": "Pressure",
    "flow_rate": "Flow Rate",
    "voltage": "Voltage",
    "current": "Current",
    "rpm": "RPM",
    "vibration": "Vibration",
    "humidity": "Humidity",
    "power_consumption": "Power Consumption",
    "response_time": "Response Time",
    "packet_size": "Packet Size",
    "command_frequency": "Command Frequency",
    "error_rate": "Error Rate",
    "network_latency": "Network Latency",
}

FEATURE_UNITS = {
    "temperature": "°C",
    "pressure": "PSI",
    "flow_rate": "L/min",
    "voltage": "V",
    "current": "A",
    "rpm": "RPM",
    "vibration": "mm/s",
    "humidity": "%",
    "power_consumption": "kW",
    "response_time": "ms",
    "packet_size": "bytes",
    "command_frequency": "cmd/min",
    "error_rate": "%",
    "network_latency": "ms",
}


def generate_explanation(params: dict, anomaly_score: float, command: str, is_anomaly: bool, hard_limits: list | None = None) -> dict:
    """Generate a detailed human-readable explanation of the prediction."""
    top_features = get_top_anomalous_features(params, top_n=4)

    flagged = []
    for feat, deviation in top_features:
        if deviation > 0.8:
            val = params.get(feat, 0)
            lo, hi = NORMAL_RANGES[feat]
            label = FEATURE_LABELS.get(feat, feat)
            unit = FEATURE_UNITS.get(feat, "")
            flagged.append({
                "feature": feat,
                "label": label,
                "value": val,
                "unit": unit,
                "normal_range": f"{lo}–{hi}",
                "deviation": deviation,
            })

    if hard_limits:
        for item in hard_limits:
            feat = item["feature"]
            label = FEATURE_LABELS.get(feat, feat)
            unit = FEATURE_UNITS.get(feat, "")
            flagged.insert(0, {
                "feature": feat,
                "label": f"{label} (Hard Limit)",
                "value": item["value"],
                "unit": unit,
                "normal_range": f"<= {item['limit']}",
                "deviation": 2.0,
            })

    if is_anomaly:
        if hard_limits:
            feat_names = ", ".join(f["label"] for f in flagged[:2])
            explanation = (
                f"⚠️ CRITICAL ANOMALY DETECTED — Hard limit exceeded for: {feat_names}. "
                f"Command '{command}' blocked for safety."
            )
            return {
                "explanation": explanation,
                "flagged_features": flagged,
                "severity": "CRITICAL",
            }

        severity = "CRITICAL" if anomaly_score > 0.8 else "HIGH" if anomaly_score > 0.6 else "MEDIUM"
        
        if len(flagged) > 0:
            feat_names = ", ".join(f["label"] for f in flagged[:3])
            explanation = (
                f"⚠️ {severity} ANOMALY DETECTED — Command '{command}' flagged with "
                f"anomaly score {anomaly_score:.2f}. Abnormal readings detected in: {feat_names}. "
            )
            for f in flagged[:2]:
                explanation += (
                    f"{f['label']} at {f['value']}{f['unit']} "
                    f"(normal: {f['normal_range']}{f['unit']}). "
                )
            explanation += "Recommend BLOCK and manual review."
        else:
            explanation = (
                f"⚠️ {severity} ANOMALY DETECTED — Command '{command}' shows anomalous patterns "
                f"with score {anomaly_score:.2f}. The combination of sensor readings deviates "
                f"from established baselines. Recommend BLOCK."
            )
    else:
        explanation = (
            f"✅ PASS — Command '{command}' evaluated with anomaly score {anomaly_score:.2f}. "
            f"All sensor readings within normal operating parameters."
        )

    return {
        "explanation": explanation,
        "flagged_features": flagged,
        "severity": "CRITICAL" if anomaly_score > 0.8 else "HIGH" if anomaly_score > 0.6 else "MEDIUM" if anomaly_score > 0.4 else "LOW",
    }
