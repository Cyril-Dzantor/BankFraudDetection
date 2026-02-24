FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy service source
COPY services/orchestrator/app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
