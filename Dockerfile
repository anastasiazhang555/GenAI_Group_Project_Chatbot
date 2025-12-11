# Use slim Python 3.11 image
FROM python:3.11-slim

# Don't write .pyc files + unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside the container
WORKDIR /app

# ---- Install system dependencies (for torch, etc.) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project code ----
COPY . .

# Expose FastAPI port
EXPOSE 8000

# ---- Start FastAPI server ----
# This uses the app.main:app 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]