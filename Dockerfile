FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only main production file and minimal assets per user preference
COPY ott_churn_pipeline.py /app/ott_churn_pipeline.py

# Create dirs
RUN mkdir -p /app/artifacts /app/data /app/sql

CMD ["python", "ott_churn_pipeline.py", "--n_users", "5000", "--n_titles", "1200"]
