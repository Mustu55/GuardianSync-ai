# GuardianSync AI Engine

FastAPI anomaly detection service.

## Deploy

Use a persistent Python host such as Render, Railway, or Fly.io.

- Install: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health: `/health`
- Configure `AI_CORS_ORIGINS` from `.env.example`

If model artifacts are absent, the service trains the bundled detector at startup. Never commit `.env` files or model artifacts.
