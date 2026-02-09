#!/usr/bin/env python3
"""
THC Main Launcher — Temporal Holographic Computation
Orchestrates engine + UI + network bridge
"""

import sys
import threading
import argparse
from pathlib import Path

from engine import THCEngine
from ui import THCControlPanel
from network import NetworkBridge
from config import HEADLESS, UI_ENABLED, VERBOSE


def main():
    parser = argparse.ArgumentParser(description="THC GPU Engine")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--no-network", action="store_true", help="Disable network bridge")
    parser.add_argument("--load", type=str, help="Load checkpoint")
    parser.add_argument("--steps", type=int, default=None, help="Max steps")

    args = parser.parse_args()

    headless = args.headless or HEADLESS

    print("=" * 60)
    print("THC GPU Engine — Temporal Holographic Computation")
    print("=" * 60)
    print()

    # Initialize engine
    try:
        engine = THCEngine()
    except Exception as e:
        print(f"[ERROR] Failed to initialize engine: {e}")
        sys.exit(1)

    # Load checkpoint if specified
    if args.load:
        try:
            engine.load_checkpoint(args.load)
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")

    # Network bridge (linked to engine for poll() integration)
    if not args.no_network:
        net = NetworkBridge(engine)
        engine.network = net  # Engine polls network in its loop
        net.start()
    else:
        net = None

    # Engine loop in background
    engine_thread = threading.Thread(
        target=lambda: engine.run_loop(max_steps=args.steps),
        daemon=False
    )
    engine_thread.start()

    # UI or headless mode
    if headless:
        print("[MAIN] Running in headless mode")
        print("[MAIN] Press Ctrl+C to stop")
        try:
            engine_thread.join()
        except KeyboardInterrupt:
            print("\n[MAIN] Stopping...")
            engine.stop()
            if net:
                net.stop()
    else:
        # Launch UI
        print("[MAIN] Launching control panel...")
        try:
            ui = THCControlPanel(engine)
            ui.run()
        except Exception as e:
            print(f"[ERROR] UI error: {e}")
        finally:
            engine.stop()
            if net:
                net.stop()

    print("[MAIN] Terminated")


if __name__ == "__main__":
    main()
