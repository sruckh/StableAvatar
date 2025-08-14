FROM runpod/pytorch:0.7.0-cu1263-torch260-ubuntu2204

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create Python 3.11 virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone the StableAvatar repository
RUN git clone https://github.com/Francis-Rings/StableAvatar.git
WORKDIR /workspace/StableAvatar

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash attention
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install huggingface_hub CLI
RUN pip install --no-cache-dir "huggingface_hub[cli]"

# Install audio-separator
RUN pip install --no-cache-dir audio-separator

# Download checkpoints
RUN mkdir -p checkpoints
RUN huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints

# Expose port for Gradio interface
EXPOSE 7860

# Set Python stream buffer to 1
ENV PYTHONUNBUFFERED=1

# Command to run the Gradio app (to be implemented)
CMD ["python", "app.py"]