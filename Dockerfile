# Dockerfile for GoldGPT Railway Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-minimal.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy essential application files only
COPY app.py .
COPY error_handling_system.py .
COPY database_config.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Create necessary directories
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the application
CMD ["python", "app.py"]
