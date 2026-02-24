FROM python:3.11-slim
WORKDIR /app

# Install dependencies (includes xgboost)
COPY services/xgboost_classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy service source
COPY services/xgboost_classifier/app.py .

# Model file is mounted at runtime via docker-compose volume
ENV PYTHONPATH=/app

EXPOSE 8004
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
