"""Pydantic schemas for AI Engine API."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    command: str = Field(..., example="SET_VALVE_OPEN")
    target_machine: str = Field(..., example="valve-01")
    industry: str = Field(default="power_plant", example="power_plant")
    parameters: dict = Field(..., example={
        "temperature": 45.2,
        "pressure": 120.5,
        "flow_rate": 85.3,
        "voltage": 230.1,
        "current": 15.2,
        "rpm": 1500,
        "vibration": 2.1,
        "humidity": 45.0,
        "power_consumption": 12.5,
        "response_time": 45,
        "packet_size": 256,
        "command_frequency": 5,
        "error_rate": 0.02,
        "network_latency": 12
    })


class PredictionResponse(BaseModel):
    action: str  # "PASS" or "BLOCK"
    anomaly_score: float
    confidence: float
    explanation: str
    flagged_features: list
    timestamp: str
    command: str
    target_machine: str


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    message: str
    samples_used: int
    model_path: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    version: str


class SimulateRequest(BaseModel):
    industry: str = "power_plant"
    mode: str = "normal"  # "normal" or "attack"
    count: int = Field(default=1, ge=1, le=100)
