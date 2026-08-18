FROM python:3.12-slim

WORKDIR /app

# Copy only dashboard requirements
COPY requirements-dashboard.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy dashboard code and paper trading data
COPY dashboard/ ./dashboard/
COPY paper_trading/ ./paper_trading/

# Expose port (Railway will set $PORT)
EXPOSE 8080

# Start the FastAPI dashboard API
CMD uvicorn dashboard.api:app --host 0.0.0.0 --port ${PORT:-8080}
