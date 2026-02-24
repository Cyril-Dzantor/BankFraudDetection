FROM python:3.11-slim
WORKDIR /app

COPY services/post_auth_intelligence/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/shared /app/shared
COPY services/post_auth_intelligence/app.py .

ENV PYTHONPATH=/app

EXPOSE 8007
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]
