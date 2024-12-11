FROM python:3.12-alpine

# Set the working directory in the container
WORKDIR /app

# Copy requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code into the container
COPY . /app

# Expose port 8000 to allow external access
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV RERANKER_TYPE=bge-m3
ENV DEVICE=cpu
ENV API_KEY="secure-api-key"


# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
