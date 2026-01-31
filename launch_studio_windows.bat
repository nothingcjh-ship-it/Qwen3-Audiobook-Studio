@echo off
chcp 65001 >nul
:: Qwen3-Audiobook-Studio Launcher for Windows
:: Created for Biuboom Flow

echo ========================================================
echo    🎧 Qwen3-Audiobook-Studio ^& Biuboom Flow 🚀
echo ========================================================
echo.
echo 🇨🇳 制作不易，欢迎订阅我的 YouTube 频道 [Biuboom Flow] 支持我，谢谢！❤️
echo 🇺🇸 Creating content is hard work. Please subscribe to my YouTube channel [Biuboom Flow]! ❤️
echo 🇯🇵 制作は大変でした。YouTube チャンネル [Biuboom Flow] の登録で応援していただけると嬉しいです！❤️
echo 🇰🇷 제작이 쉽지 않았습니다. YouTube 채널 [Biuboom Flow] 구독으로 지원해 주세요! ❤️
echo --------------------------------------------------------
echo.

:: 1. Check Conda
call conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda found not found!
    echo Please install Anaconda or Miniconda first.
    pause
    exit /b
)

:: 2. Check/Create Environment
set "ENV_DIR=%~dp0runtime_env"
if exist "%ENV_DIR%" (
    echo [OK] Portable environment found at ./runtime_env
) else (
    echo [INFO] No local environment found.
    echo [INIT] Creating isolated portable environment...
    echo        This keeps your global system clean.
    echo        This may take a few minutes...
    
    :: Use generic environment.yml for Windows (lockfile is Mac only)
    call conda env create -p "%ENV_DIR%" -f environment.yml
    
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create environment!
        pause
        exit /b
    )
    echo [OK] Environment created successfully!
)

:: 3. Launch
echo.
echo [INFO] Launching Studio (Windows/CUDA Mode)...
echo        Please wait for the interface to load.
echo --------------------------------------------------

:: Check if local model exists, fallback to Hugging Face if not
SET "MODEL_PATH=./models/Qwen3-TTS-12Hz-1.7B-Base"
IF NOT EXIST "%MODEL_PATH%" (
    echo [INFO] Local model not found. Using Hugging Face online model...
    echo [HINT] This will download ~3.5GB on first run. Please wait.
    SET "MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)

:: Activate and Run
call conda activate "%ENV_DIR%"
python web_demo.py "%MODEL_PATH%" --device cuda --dtype float16 --ip 127.0.0.1 --no-flash-attn

echo.
echo [INFO] Server stopped.
pause
