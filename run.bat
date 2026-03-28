@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo [检查环境 / Checking environment...]

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python 未安装或未在 PATH 中。请安装 Python 3.8+。
    echo [Error] Python is not installed or not in PATH.
    pause
    exit /b
)

REM Check if venv exists
if not exist "venv" (
    echo [创建虚拟环境 / Creating virtual environment...]
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败。
        echo [Error] Failed to create virtual environment.
        pause
        exit /b
    )
    
    echo [激活虚拟环境 / Activating virtual environment...]
    call venv\Scripts\activate
    
    echo [安装依赖 / Installing dependencies...]
    echo 这可能需要一些时间，请耐心等待...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 安装依赖失败。
        echo [Error] Failed to install dependencies.
        pause
        exit /b
    )
) else (
    echo [激活虚拟环境 / Activating virtual environment...]
    call venv\Scripts\activate
)

REM Check for .env file
if not exist ".env" (
    echo [警告] 未找到 .env 配置文件。可能需要配置 API Key。
    echo [Warning] .env file not found.
)

REM Run the application
echo [启动应用 / Starting Application...]
streamlit run app.py

pause
