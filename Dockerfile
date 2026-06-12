# TensorFlow GPU image for CUDA support
FROM tensorflow/tensorflow:2.15.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.server.txt

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output

# HF Token as build argument (optional, can be passed at build or runtime)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Set environment variables for GPU
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command
CMD ["python3", "main.py", "--mode", "fast", "--generations", "10"]
