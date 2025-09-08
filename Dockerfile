# Use a base image with CUDA 12.4 and Python 3.10
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and other system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up the working directory
RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser/app

# Copy the application code into the container
# We will fix permissions in a later step
COPY . /home/appuser/app/audiocraft
COPY ./wheels /home/appuser/app/wheels
COPY main.py /home/appuser/app/

# --- FIX: Change ownership of all copied files to the appuser ---
RUN chown -R appuser:appuser /home/appuser/app

# Switch to the non-root user for all subsequent commands
USER appuser

# --- FIX: Add the user's local bin to the PATH ---
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install Python dependencies from local wheels and PyPI
RUN pip3 install --no-cache-dir /home/appuser/app/wheels/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl
RUN pip3 install torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# --- FIX: Install transformers WITH its dependencies, and add missing ones ---
RUN pip3 install --no-cache-dir transformers==4.38.0 accelerate==0.20.3 safetensors==0.4.3 tokenizers==0.15.2 \
    huggingface-hub requests tqdm psutil regex

RUN pip3 install --no-cache-dir xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124 --no-deps
RUN pip3 install --no-cache-dir av==11.0.0 julius==0.2.7 flashy==0.0.2 num2words==0.5.14 torchdiffeq==0.2.5 torchmetrics==1.8.1

# Install the audiocraft package itself (this should now succeed)
RUN cd /home/appuser/app/audiocraft && pip3 install --no-cache-dir -e . --no-deps

# Install FastAPI and Uvicorn
RUN pip3 install --no-cache-dir fastapi uvicorn[standard] gradio

# Expose the port Vertex AI expects
EXPOSE 8080

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
