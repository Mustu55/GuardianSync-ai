"""Normal SCADA sensor stream generator for different industry types."""

import random
import numpy as np
from .utils import NORMAL_RANGES

INDUSTRY_PROFILES = {
    "power_plant": {
        "machines": ["turbine-01", "generator-01", "transformer-01", "cooling-tower-01", "boiler-01"],
        "commands": ["START_TURBINE", "SET_POWER_OUTPUT", "ADJUST_COOLING", "CHECK_VOLTAGE", "SYNC_GENERATOR"],
        "temp_bias": 15.0,
        "pressure_bias": 30.0,
    },
    "water_treatment": {
        "machines": ["intake-pump-01", "filter-unit-01", "chlorinator-01", "storage-tank-01", "dist-pump-01"],
        "commands": ["START_PUMP", "SET_FLOW_RATE", "ADD_CHLORINE", "CHECK_PH", "FLUSH_FILTER"],
        "temp_bias": -5.0,
        "pressure_bias": -20.0,
    },
    "manufacturing": {
        "machines": ["conveyor-01", "robot-arm-01", "press-01", "qc-scanner-01", "packaging-01"],
        "commands": ["START_CONVEYOR", "MOVE_ARM", "ACTIVATE_PRESS", "RUN_INSPECTION", "SEAL_PACKAGE"],
        "temp_bias": 5.0,
        "pressure_bias": 10.0,
    },
    "chemical_facility": {
        "machines": ["reactor-01", "distiller-01", "mixer-01", "storage-vessel-01", "scrubber-01"],
        "commands": ["START_REACTION", "SET_TEMPERATURE", "ADJUST_MIX_RATIO", "VENT_PRESSURE", "ACTIVATE_SCRUBBER"],
        "temp_bias": 25.0,
        "pressure_bias": 50.0,
    },
}


def generate_normal_reading(industry: str = "power_plant") -> dict:
    """Generate a single normal sensor reading for the given industry."""
    profile = INDUSTRY_PROFILES.get(industry, INDUSTRY_PROFILES["power_plant"])
    params = {}
    for feat, (lo, hi) in NORMAL_RANGES.items():
        mid = (lo + hi) / 2
        std = (hi - lo) / 6  # 99.7% within range
        val = np.random.normal(mid, std)
        val = max(lo, min(hi, val))
        # Apply industry-specific bias for temp/pressure
        if feat == "temperature":
            val += profile.get("temp_bias", 0)
        elif feat == "pressure":
            val += profile.get("pressure_bias", 0)
        params[feat] = round(val, 2)

    return {
        "command": random.choice(profile["commands"]),
        "target_machine": random.choice(profile["machines"]),
        "industry": industry,
        "parameters": params,
    }


def generate_normal_batch(industry: str = "power_plant", count: int = 100) -> list:
    """Generate a batch of normal readings."""
    return [generate_normal_reading(industry) for _ in range(count)]
