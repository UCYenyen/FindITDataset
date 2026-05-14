# FindIT Backend API

FastAPI backend wrapping the trained Prophet + LightGBM + Isolation Forest hybrid model. Designed to be consumed by the Next.js dashboard.

## Quick Start (Local)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# from project root, run:
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Server + model status |
| GET | `/api/metrics` | MAE / RMSE / MAPE / R² |
| GET | `/api/forecast/historical?start=&end=` | Historical actual vs predicted (CSV-backed) |
| GET | `/api/forecast/future?days=7` | N-day-ahead forecast (live inference) |
| POST | `/api/forecast/whatif` | What-if scenario + SHAP waterfall |
| GET | `/api/anomalies` | Isolation Forest anomalies |
| GET | `/api/features` | All 18 features with metadata |
| GET | `/api/features/required` | Just user-supplied fields for what-if |
| GET | `/api/features/importance` | Global SHAP importance |

## What-If Request Example

```bash
curl -X POST http://localhost:8000/api/forecast/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-06-01",
    "avg_temp": 28.5,
    "rainfall": 5.2,
    "is_holiday": false
  }'
```

Response includes `predicted_mwh`, `prophet_baseline`, `lgbm_residual`, `base_value`, and a sorted list of `shap_contributions` (ready for waterfall chart).

## Next.js Integration

```typescript
// lib/api.ts
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export async function getMetrics() {
  const res = await fetch(`${API}/metrics`);
  return res.json();
}

export async function getHistorical(start?: string, end?: string) {
  const qs = new URLSearchParams();
  if (start) qs.set("start", start);
  if (end) qs.set("end", end);
  const res = await fetch(`${API}/forecast/historical?${qs}`);
  return res.json();
}

export async function getFuture(days = 7) {
  const res = await fetch(`${API}/forecast/future?days=${days}`);
  return res.json();
}

export async function runWhatIf(payload: {
  target_date: string;
  avg_temp: number;
  rainfall: number;
  is_holiday: boolean;
}) {
  const res = await fetch(`${API}/forecast/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function getAnomalies() {
  const res = await fetch(`${API}/anomalies`);
  return res.json();
}
```

## Deploying

### Railway / Render / Fly.io

The `backend/Dockerfile` is in the project root. Build context must be the **repo root** so it can copy `Models/` and `Outputs/`:

```bash
docker build -f backend/Dockerfile -t findit-api .
docker run -p 8000:8000 -e CORS_ORIGINS="https://your-frontend.vercel.app" findit-api
```

Set `CORS_ORIGINS` in production to your Next.js domain.

## Architecture

```
backend/
├── app/
│   ├── main.py           # FastAPI app, CORS, router registration
│   ├── config.py         # Paths + env config
│   ├── model_store.py    # Singleton: loads .joblib once at startup
│   ├── schemas.py        # Pydantic request/response models
│   ├── routes/           # Thin HTTP layer
│   └── services/         # Business logic (forecast, metrics, shap, anomalies)
├── Dockerfile
└── requirements.txt
```

Models load **once** at startup via FastAPI's `lifespan`. Historical reads from `Outputs/dataset_daily_with_predictions.csv`. What-if + future forecasts run live inference using the loaded models. SHAP uses cached `TreeExplainer`.
