FROM runpod/pytorch:0.7.0-cu1263-torch260-ubuntu2204

# Set working directory
WORKDIR /workspace

# Copy runtime setup script and the custom Gradio application files
COPY setup_runtime.sh /workspace/setup_runtime.sh
COPY app.py /workspace/app.py
COPY audio_extractor.py /workspace/audio_extractor.py
COPY vocal_seperator.py /workspace/vocal_seperator.py

# Make the setup script executable
RUN chmod +x /workspace/setup_runtime.sh

# Expose port for Gradio interface
EXPOSE 7860

# Set Python stream buffer to 1
ENV PYTHONUNBUFFERED=1

# Use the runtime setup script as entry point.
# This script will clone the repo, move the app files into it,
# install dependencies, and then execute the CMD.
ENTRYPOINT ["/workspace/setup_runtime.sh"]
CMD ["python", "app.py"]