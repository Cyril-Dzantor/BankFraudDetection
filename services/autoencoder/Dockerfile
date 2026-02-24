FROM python:3.11-slim
WORKDIR /app

# Install dependencies (includes scikit-learn for Autoencoder)
COPY services/autoencoder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy service source
COPY services/autoencoder/app.py .

# Model files are mounted at runtime via docker-compose volume
ENV PYTHONPATH=/app

EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
