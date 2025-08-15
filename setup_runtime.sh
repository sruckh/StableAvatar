#!/bin/bash

# Runtime setup script for StableAvatar on Runpod
# This script runs at RUNTIME (not part of container build)

set -e  # Exit on any error

echo "Starting StableAvatar runtime setup..."

# 0) Install ffmpeg from OS packages (if not already installed)
echo "Installing ffmpeg..."
apt-get update && apt-get install -y ffmpeg

# 1) Python Stream buffer set to 1
export PYTHONUNBUFFERED=1
echo "Python stream buffer set to 1"

# 2) Create Python 3.11 virtual environment
echo "Creating Python 3.11 virtual environment..."
python3.11 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Virtual environment activated. Python location: $(which python)"

# 3) Clone repo https://github.com/Francis-Rings/StableAvatar.git
echo "Cloning StableAvatar repository..."
cd /workspace
# Ensure we start with a fresh clone each time
rm -rf StableAvatar
git clone https://github.com/Francis-Rings/StableAvatar.git

# Move custom application files into the cloned repository
echo "Moving custom application files into the repository..."
mv /workspace/app.py /workspace/StableAvatar/app.py
mv /workspace/audio_extractor.py /workspace/StableAvatar/audio_extractor.py
mv /workspace/vocal_seperator.py /workspace/StableAvatar/vocal_seperator.py

cd StableAvatar

# 4) Change to StableAvatar directory (already done above)

# 5) pip install -r requirements.txt
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 6) Install flash attention from source to ensure ABI compatibility
echo "Installing flash attention from source..."
pip install flash-attn --no-build-isolation

# 7) pip install "huggingface_hub[cli]"
echo "Installing HuggingFace Hub CLI..."
pip install "huggingface_hub[cli]"

# 8) mkdir checkpoints
echo "Creating checkpoints directory..."
mkdir -p checkpoints

# 9) huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
echo "Installing hf-transfer for faster downloads..."
pip install hf-transfer
echo "Downloading model checkpoints..."
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints

# 10) All dependencies are now handled by requirements.txt

echo "Runtime setup completed successfully!"
echo "To run the Gradio interface:"
echo "1. Activate the virtual environment: source /workspace/venv/bin/activate"
echo "2. Navigate to the project directory: cd /workspace/StableAvatar"
echo "3. Run the app: python app.py"

# Print environment info for debugging
echo "--- Printing environment versions for debugging ---"
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
echo "-------------------------------------------------"

# Keep the container running
exec "$@"