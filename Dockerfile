FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first to keep the image slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python libraries
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /workspace/input /workspace/models /workspace/results

# Default command
CMD ["bash"]
