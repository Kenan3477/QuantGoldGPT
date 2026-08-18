FROM python:3.12-slim

WORKDIR /app

# Copy only dashboard requirements
COPY requirements-dashboard.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy dashboard code
COPY dashboard/ ./dashboard/

# Copy paper trading data if it exists (won't fail if directory is missing)
COPY --chown=root:root paper_trading ./paper_trading

# Make startup script executable
RUN chmod +x /app/dashboard/startup.sh

# Expose port (Railway will set $PORT)
EXPOSE 8080

# Use startup script for better logging
CMD ["/bin/bash", "/app/dashboard/startup.sh"]
