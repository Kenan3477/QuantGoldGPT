# Dockerfile for GoldGPT Railway Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Expose port (Railway sets PORT dynamically)
EXPOSE $PORT

# Use our production WSGI server
CMD ["sh", "-c", "gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:${PORT:-5000} wsgi:application"]
