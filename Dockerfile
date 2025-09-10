FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all required Python files explicitly
COPY app.py .
COPY signal_memory_system.py .
COPY real_pattern_detection.py .
COPY templates/ templates/
COPY static/ static/

EXPOSE $PORT

CMD ["python", "app.py"]
