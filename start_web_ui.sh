#!/bin/bash
# Activate environment and run web demo
# Ensure we are in the script's directory
cd "$(dirname "$0")"

echo "Starting Qwen3-TTS Web UI..."
echo "Please wait for the model to load..."
echo "Once loaded, you can access the interface at: http://127.0.0.1:8000"

# Use conda run to ensure environment is active
# Note: we use -u for unbuffered output so you can see logs immediately
conda run -n qwen3-tts python -u web_demo.py ./models/Qwen3-TTS-12Hz-1.7B-Base --device mps --dtype bfloat16 --ip 127.0.0.1 --no-flash-attn
