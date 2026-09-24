"""Attack payload generator for SCADA anomaly simulation."""

import random
import numpy as np
from .utils import NORMAL_RANGES
from .simulator import INDUSTRY_PROFILES

ATTACK_TYPES = {
    "spike": "Sudden value spike beyond safe operating range",
    "flatline": "Sensor reading dropped to zero or near-zero",
    "oscillation": "Rapid oscillation indicating sensor tampering",
    "overflow": "Command frequency flooding (DoS-style)",
    "drift": "Gradual drift outside normal operating envelope",
}


def _apply_spike(params: dict) -> dict:
    targets = random.sample(list(NORMAL_RANGES.keys()), k=random.randint(2, 5))
    for feat in targets:
        lo, hi = NORMAL_RANGES[feat]
        params[feat] = round(hi * random.uniform(2.5, 5.0), 2)
    return params


def _apply_flatline(params: dict) -> dict:
    targets = random.sample(list(NORMAL_RANGES.keys()), k=random.randint(2, 4))
    for feat in targets:
        params[feat] = round(random.uniform(0, 0.01), 4)
    return params


def _apply_oscillation(params: dict) -> dict:
    targets = random.sample(list(NORMAL_RANGES.keys()), k=random.randint(2, 4))
    for feat in targets:
        lo, hi = NORMAL_RANGES[feat]
        params[feat] = round(random.choice([lo * 0.1, hi * 3.0, lo * 0.5, hi * 2.5]), 2)
    params["vibration"] = round(random.uniform(15.0, 50.0), 2)
    return params


def _apply_overflow(params: dict) -> dict:
    params["command_frequency"] = round(random.uniform(50, 200), 1)
    params["packet_size"] = round(random.uniform(2048, 8192), 0)
    params["network_latency"] = round(random.uniform(200, 1000), 1)
    params["error_rate"] = round(random.uniform(0.3, 0.9), 3)
    return params


def _apply_drift(params: dict) -> dict:
    targets = random.sample(list(NORMAL_RANGES.keys()), k=random.randint(3, 6))
    for feat in targets:
        lo, hi = NORMAL_RANGES[feat]
        direction = random.choice([-1, 1])
        drift_amount = (hi - lo) * random.uniform(0.8, 1.5)
        params[feat] = round(params[feat] + direction * drift_amount, 2)
    return params


ATTACK_FUNCTIONS = {
    "spike": _apply_spike,
    "flatline": _apply_flatline,
    "oscillation": _apply_oscillation,
    "overflow": _apply_overflow,
    "drift": _apply_drift,
}


def generate_attack_reading(industry: str = "power_plant", attack_type: str = None) -> dict:
    """Generate a single attack payload for the given industry."""
    from .simulator import generate_normal_reading
    reading = generate_normal_reading(industry)

    if attack_type is None:
        attack_type = random.choice(list(ATTACK_TYPES.keys()))

    attack_fn = ATTACK_FUNCTIONS[attack_type]
    reading["parameters"] = attack_fn(reading["parameters"])

    # Override command to something suspicious
    suspicious_commands = [
        "EMERGENCY_SHUTDOWN", "OVERRIDE_SAFETY", "DISABLE_ALARM",
        "FORCE_VALVE_OPEN", "BYPASS_INTERLOCK", "SET_MAX_PRESSURE",
        "DISABLE_COOLING", "OVERRIDE_TEMP_LIMIT", "FORCE_RESTART"
    ]
    reading["command"] = random.choice(suspicious_commands)
    reading["_attack_type"] = attack_type
    reading["_attack_desc"] = ATTACK_TYPES[attack_type]

    return reading


def generate_attack_batch(industry: str = "power_plant", count: int = 50) -> list:
    """Generate a batch of attack readings of mixed types."""
    return [generate_attack_reading(industry) for _ in range(count)]
