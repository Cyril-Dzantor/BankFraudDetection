FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY services/rules_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schemas
COPY services/shared /app/shared

# Copy project source modules needed by the Rules Engine
COPY Rules\ Engine /app/Rules Engine
COPY Decision\ Layer /app/Decision Layer

# Copy service source
COPY services/rules_engine/app.py .

ENV PYTHONPATH=/app

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
