# Use the official Python image
FROM python:3.9-slim
LABEL authors="Laney_Black"

# Set working directory
WORKDIR /app

# Copy the backend project files into the container
COPY backend/ /app/

# Ensure Python and pip are installed
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required Python dependencies
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# If model.safetensors doesn't exist, run construction.py to create it
RUN if [ ! -f "/app/ml_model/complete_model/model.safetensors" ]; then \
    echo "Model file not found. Running construction script..."; \
    python /app/ml_model/construction.py; \
  fi

# Expose the port for FastAPI
EXPOSE 10000

# Start FastAPI app
CMD ["python", "app.py"]
# Or the command that runs your FastAPI app

ENTRYPOINT ["top", "-b"]