FROM docker.io/library/python:3.10-slim

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY infrang_core.py infrang-api.py ./

# Expose the port
EXPOSE 7456

# Run the application
CMD ["uvicorn", "infrang-api:app", "--host", "0.0.0.0", "--port", "7456"]