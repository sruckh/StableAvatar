FROM runpod/pytorch:0.7.0-cu1263-torch260-ubuntu2204

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy runtime setup script
COPY setup_runtime.sh /workspace/setup_runtime.sh
RUN chmod +x /workspace/setup_runtime.sh

# Copy application files
COPY . /workspace/StableAvatar
WORKDIR /workspace/StableAvatar

# Expose port for Gradio interface
EXPOSE 7860

# Set Python stream buffer to 1
ENV PYTHONUNBUFFERED=1

# Use the runtime setup script as entry point
ENTRYPOINT ["/workspace/setup_runtime.sh"]
CMD ["python", "app.py"]