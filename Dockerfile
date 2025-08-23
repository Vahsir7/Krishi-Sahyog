# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (needed for FAISS, sqlite, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Copy environment file
COPY .env .env

# Cloud Run expects the app to listen on $PORT
ENV PORT=8080

EXPOSE 8080


# Start FastAPI using uvicorn
CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
