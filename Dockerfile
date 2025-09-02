# Dockerfile for GoldGPT Railway Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-ultra-minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-ultra-minimal.txt

# Copy minimal application files only
COPY minimal_app.py .
COPY emergency_signal_generator.py .

# Create necessary directories
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application with gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 minimal_app:app"]
