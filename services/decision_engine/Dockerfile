FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY services/decision_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy Decision Layer source + config
COPY Decision\ Layer /app/Decision Layer

# Copy service source
COPY services/decision_engine/app.py .

ENV PYTHONPATH=/app

EXPOSE 8005
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
