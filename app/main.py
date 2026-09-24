"""FastAPI application — GuardianSync AI Engine."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import logging
import os

from dotenv import load_dotenv

from .schemas import (
    PredictionRequest, PredictionResponse,
    TrainResponse, HealthResponse, SimulateRequest
)
from .inference import predict, is_model_loaded, load_model
from .explain import generate_explanation
from .train import train_model
from .simulator import generate_normal_reading, generate_normal_batch
from .attack_generator import generate_attack_reading, generate_attack_batch

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GuardianSync AI Engine",
    version="1.0.0",
    description="SCADA anomaly detection and simulation engine"
)

cors_origins = [o.strip() for o in os.getenv("AI_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
allow_any_origin = "*" in cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_any_origin else cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request error: {request.method} {request.url.path} - {str(e)}")
        raise

# Try to load model on startup
@app.on_event("startup")
async def startup():
    try:
        if not load_model():
            logger.info("Model artifacts not found; training the bundled detector")
            train_model()
            load_model()
        logger.info("AI model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model on startup: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        model_loaded = is_model_loaded()
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(req: PredictionRequest):
    try:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        # Validate input
        if not req.parameters or not isinstance(req.parameters, dict):
            raise HTTPException(status_code=400, detail="Invalid parameters")
        if not req.command or not isinstance(req.command, str):
            raise HTTPException(status_code=400, detail="Invalid command")
        if not req.target_machine or not isinstance(req.target_machine, str):
            raise HTTPException(status_code=400, detail="Invalid target_machine")
        
        # Predict
        result = predict(req.parameters)

        if "error" in result:
            logger.warning(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=503, detail=result["error"])

        # Generate explanation
        explanation_data = generate_explanation(
            params=req.parameters,
            anomaly_score=result["anomaly_score"],
            command=req.command,
            is_anomaly=result["is_anomaly"],
            hard_limits=result.get("hard_limits")
        )

        action = "BLOCK" if result["is_anomaly"] else "PASS"
        # Improved confidence: higher when score is more extreme (close to 0 or 1)
        anomaly_score = result["anomaly_score"]
        confidence = max(0.5, min(1.0, 1.0 - abs(anomaly_score - 0.5)))

        return PredictionResponse(
            action=action,
            anomaly_score=round(anomaly_score, 4),
            confidence=round(confidence, 3),
            explanation=explanation_data["explanation"],
            flagged_features=explanation_data["flagged_features"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            command=req.command,
            target_machine=req.target_machine,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/train", response_model=TrainResponse)
async def train_endpoint():
    try:
        result = train_model()
        load_model()
        logger.info(f"Model trained successfully with {result['samples_used']} samples")
        return TrainResponse(
            status=result["status"],
            message=result["message"],
            samples_used=result["samples_used"],
            model_path=result["model_path"]
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail="Training failed")


@app.post("/simulate")
async def simulate_endpoint(req: SimulateRequest):
    try:
        if req.count < 1 or req.count > 100:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 100")
        
        readings = []
        if req.mode == "attack":
            readings = generate_attack_batch(req.industry, req.count)
        else:
            readings = generate_normal_batch(req.industry, req.count)
        
        return {"readings": readings, "count": len(readings), "mode": req.mode}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Simulation failed")


@app.get("/industries")
async def list_industries():
    try:
        from .simulator import INDUSTRY_PROFILES
        return {
            "industries": list(INDUSTRY_PROFILES.keys()),
            "profiles": {
                k: {"machines": v["machines"], "commands": v["commands"]}
                for k, v in INDUSTRY_PROFILES.items()
            }
        }
    except Exception as e:
        logger.error(f"Industries list error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list industries")
