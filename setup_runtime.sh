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
if [ ! -d "StableAvatar" ]; then
    git clone https://github.com/Francis-Rings/StableAvatar.git
fi
cd StableAvatar

# 4) Change to StableAvatar directory (already done above)

# 5) pip install -r requirements.txt
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 6) Install flash attention wheel
echo "Installing flash attention..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 7) pip install "huggingface_hub[cli]"
echo "Installing HuggingFace Hub CLI..."
pip install "huggingface_hub[cli]"

# 8) mkdir checkpoints
echo "Creating checkpoints directory..."
mkdir -p checkpoints

# 9) huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints
echo "Downloading model checkpoints..."
huggingface-cli download FrancisRing/StableAvatar --local-dir ./checkpoints

# 10) pip install audio-separator
echo "Installing audio-separator..."
pip install audio-separator

echo "Runtime setup completed successfully!"
echo "To run the Gradio interface:"
echo "1. Activate the virtual environment: source /workspace/venv/bin/activate"
echo "2. Navigate to the project directory: cd /workspace/StableAvatar"
echo "3. Run the app: python app.py"

# Keep the container running
exec "$@"