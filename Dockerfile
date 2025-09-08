# Use a base image with CUDA 12.4 and Python 3.10
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and other system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it for security
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the audiocraft repository into the container
COPY . /home/appuser/app/audiocraft

# Copy the local wheels for installation
COPY ./wheels /home/appuser/app/wheels

# Install Python dependencies from local wheels and PyPI
RUN pip3 install --no-cache-dir /home/appuser/app/wheels/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl
RUN pip3 install torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir transformers==4.38.0 accelerate==0.20.3 safetensors==0.4.3 tokenizers==0.15.2 --no-deps
RUN pip3 install --no-cache-dir xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124 --no-deps
RUN pip3 install --no-cache-dir av==11.0.0 julius==0.2.7 flashy==0.0.2 num2words==0.5.14 torchdiffeq==0.2.5 torchmetrics==1.8.1

# Install the audiocraft package itself
RUN cd /home/appuser/app/audiocraft && pip3 install --no-cache-dir -e . --no-deps

# Install FastAPI and Uvicorn
RUN pip3 install --no-cache-dir fastapi uvicorn[standard]

# Copy the new main.py script
COPY main.py /home/appuser/app/

# Expose the port Vertex AI expects
EXPOSE 8080

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
