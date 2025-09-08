# Use a base image with CUDA and Python that matches the requirements
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10, pip, and git
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up the working directory
RUN useradd -ms /bin/bash appuser
WORKDIR /home/appuser/app

# Copy all the necessary files
COPY . /home/appuser/app/audiocraft
COPY requirements.txt /home/appuser/app/
COPY main.py /home/appuser/app/

# Change ownership of all copied files to the appuser
RUN chown -R appuser:appuser /home/appuser/app

# Switch to the non-root user for all subsequent commands
USER appuser

# Add the user's local bin to the PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install the exact required versions of torch, torchvision, and torchaudio first
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies from your requirements.txt file
RUN pip3 install --no-cache-dir -r /home/appuser/app/requirements.txt

# Install the audiocraft package itself in editable mode
RUN cd /home/appuser/app/audiocraft && pip3 install --no-cache-dir -e .

# Install FastAPI and Uvicorn for the web server
RUN pip3 install --no-cache-dir fastapi uvicorn[standard]

# Expose the port Vertex AI expects
EXPOSE 8080

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
