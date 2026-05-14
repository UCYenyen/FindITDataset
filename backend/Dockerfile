FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY Models ./Models
COPY Outputs ./Outputs

ENV MODELS_DIR=/app/Models
ENV OUTPUTS_DIR=/app/Outputs
ENV PORT=8000

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
