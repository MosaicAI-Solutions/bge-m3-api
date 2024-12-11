FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# # Install build dependencies
# RUN apk add --no-cache gcc g++ musl-dev make python3-dev libffi-dev cmake ninja pkgconfig
# # RUN pip cache purge

# RUN pip install --no-cache-dir jinja2==3.1.2 MarkupSafe==2.0
# # RUN pip install --no-cache-dir sentencepiece==0.2.0 pyarrow==12.0.1
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && \
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
