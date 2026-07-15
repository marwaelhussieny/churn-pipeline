FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY serving/ ./serving/
COPY src/mlflow.db ./src/mlflow.db
COPY src/mlruns ./src/mlruns

ENV MLFLOW_TRACKING_URI=sqlite:////app/src/mlflow.db
ENV REDIS_HOST=redis
ENV PYTHONPATH=/app/serving

EXPOSE 8000

CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
