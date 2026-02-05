#!/bin/bash

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================================"
echo "   🎧 Qwen3-Audiobook-Studio & Biuboom Flow 🚀"
echo "========================================================"
echo ""
echo "🇨🇳 制作不易，欢迎订阅我的 YouTube 频道 [Biuboom Flow] 支持我，谢谢！❤️"
echo "🇺🇸 Creating content is hard work. Please subscribe to my YouTube channel [Biuboom Flow]! ❤️"
echo "🇯🇵 制作は大変でした。YouTube チャンネル [Biuboom Flow] の登録で応援していただけると嬉しいです！❤️"
echo "🇰🇷 제작이 쉽지 않았습니다. YouTube 채널 [Biuboom Flow] 구독으로 지원해 주세요! ❤️"
echo "--------------------------------------------------------"
echo ""

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# 2. Check for Local Environment
ENV_DIR="$DIR/runtime_env"

if [ -d "$ENV_DIR" ]; then
    echo "[OK] Portable environment found at ./runtime_env"
else
    echo "[INFO] No local environment found."
    echo "[INIT] Creating isolated portable environment in ./runtime_env..."
    echo "       This keeps your global system clean."
    echo "       This process may take a few minutes..."
    
    # Strategy: Create bare Python env (fast) -> Use Pip (fast)
    # This avoids Conda's slow "Solving environment" step completely.
    
    # 2.1 Ask for Region (Mirror Selection)
    echo "--------------------------------------------------"
    echo "🌍 Select your region for faster download / 选择您的地区以加速下载:"
    echo "1. Global / 全球 (Default)"
    echo "2. China  / 中国 (Tsinghua Mirror / 清华源)"
    read -p "Enter 1 or 2: " region_choice
    echo "--------------------------------------------------"

    PIP_MIRROR=""
    if [ "$region_choice" == "2" ]; then
        echo "[INFO] Using Tsinghua Mirror for Pip..."
        PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
    
    echo "[INIT] Creating base Python 3.10 environment with ffmpeg..."
    conda create -p "$ENV_DIR" python=3.10 ffmpeg -y
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create environment!"
        exit 1
    fi
    
    echo "[INSTALL] Installing dependencies via Pip..."
    "$ENV_DIR/bin/pip" install --upgrade pip $PIP_MIRROR
    "$ENV_DIR/bin/pip" install -r requirements.txt $PIP_MIRROR
    
    if [ $? -ne 0 ]; then
         echo "[ERROR] Pip install failed."
         exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create environment!"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Environment created successfully!"
fi

# 3. Launch
echo ""
echo "[INFO] Launching Studio..."
echo "       Using environment: $ENV_DIR"
echo "--------------------------------------------------"

# Check if local model exists, fallback to Hugging Face if not
MODEL_PATH="./models/Qwen3-TTS-12Hz-1.7B-Base"
if [ ! -d "$MODEL_PATH" ]; then
    echo "[INFO] Local model not found. Using Hugging Face online model..."
    echo "[HINT] This will download ~3.5GB on first run. Please wait."
    MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
fi

# Run directly using the environment's python executable
# This is more robust than 'conda run' which can silence errors or exit unexpectedly
"$ENV_DIR/bin/python" web_demo.py "$MODEL_PATH" --device mps --dtype bfloat16 --ip 127.0.0.1 --no-flash-attn

echo ""
echo "[INFO] Server stopped."
read -p "Press Enter to close..."
