@echo off
REM THC Batch Launcher for Windows — Temporal Holographic Computation

setlocal enabledelayedexpansion

echo.
echo ========================================
echo THC GPU Engine Launcher (Windows)
echo ========================================
echo.

if "%1"=="test" (
    echo Running test suite...
    python test.py
    goto end
)

if "%1"=="headless" (
    echo Running in headless mode (Ctrl+C to stop)...
    python main.py --headless
    goto end
)

if "%1"=="no-net" (
    echo Running without network bridge...
    python main.py --no-network
    goto end
)

if "%1"=="load" (
    if "%2"=="" (
        echo Usage: run.bat load ^<checkpoint_file^>
        goto end
    )
    echo Loading checkpoint: %2
    python main.py --load "%2"
    goto end
)

REM Default: UI mode
echo Running with control panel...
python main.py

:end
endlocal
