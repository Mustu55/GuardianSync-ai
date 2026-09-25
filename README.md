# GuardianSync AI Engine

FastAPI anomaly detection service.

## Deploy on Render

Create a Python **Web Service** from this repository. The included `render.yaml` provides the deployment settings.

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Python runtime: `3.11.9`
- Configure `AI_CORS_ORIGINS` with the deployed client and server URLs, separated by commas.

If model artifacts are absent, the service trains the bundled detector at startup. Never commit `.env` files or model artifacts.

After deployment, set the backend's `AI_ENGINE_URL` to the AI service base URL, without `/api`, `/predict`, or `/health`.
