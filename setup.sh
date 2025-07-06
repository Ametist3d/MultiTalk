#!/bin/bash
# System dependencies
apt update
apt install -y ffmpeg nano vim curl wget

# Python dependencies  
pip install -r requirements.txt

echo "Setup complete!"