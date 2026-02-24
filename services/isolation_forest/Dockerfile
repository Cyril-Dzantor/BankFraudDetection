FROM python:3.11-slim
WORKDIR /app

# Install dependencies (includes scikit-learn for Isolation Forest)
COPY services/isolation_forest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy service source
COPY services/isolation_forest/app.py .

# Model file is mounted at runtime via docker-compose volume
ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
