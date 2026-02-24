FROM python:3.11-slim
WORKDIR /app

COPY services/action_layer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/shared /app/shared
COPY services/action_layer/app.py .

ENV PYTHONPATH=/app

EXPOSE 8006
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8006"]
