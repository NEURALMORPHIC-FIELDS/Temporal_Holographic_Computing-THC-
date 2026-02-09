#!/bin/bash
# THC Launcher for Linux/macOS — Temporal Holographic Computation

echo ""
echo "========================================"
echo "THC GPU Engine Launcher (Linux/macOS)"
echo "========================================"
echo ""

case "$1" in
    test)
        echo "Running test suite..."
        python3 test.py
        ;;
    headless)
        echo "Running in headless mode (Ctrl+C to stop)..."
        python3 main.py --headless
        ;;
    no-net)
        echo "Running without network bridge..."
        python3 main.py --no-network
        ;;
    load)
        if [ -z "$2" ]; then
            echo "Usage: ./run.sh load <checkpoint_file>"
        else
            echo "Loading checkpoint: $2"
            python3 main.py --load "$2"
        fi
        ;;
    *)
        echo "Running with control panel..."
        python3 main.py
        ;;
esac
