#!/usr/bin/env python3
"""
THC Visualization — Temporal Holographic Computation
Live heatmap of FHIDS field
"""

import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from collections import deque
from pathlib import Path

from config import P, D

# Max data points to keep in history
_HISTORY_LEN = 500


class THCVisualizer:
    def __init__(self, engine, update_interval=0.1):
        """Initialize live visualization."""
        self.engine = engine
        self.update_interval = update_interval

        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle("THC Holographic Field Monitor", fontsize=14, fontweight='bold')

        # Subplots
        self.ax_field = self.axes[0, 0]
        self.ax_pins = self.axes[0, 1]
        self.ax_energy = self.axes[1, 0]
        self.ax_entropy = self.axes[1, 1]

        # History buffers (bounded deques - O(1) append/discard)
        self.energy_history = deque(maxlen=_HISTORY_LEN)
        self.entropy_history = deque(maxlen=_HISTORY_LEN)
        self.step_history = deque(maxlen=_HISTORY_LEN)

        # Initialize persistent plot elements to avoid recreating every frame
        self._init_plots()

        plt.tight_layout()
        plt.ion()

        print("[VISUALIZER] Initialized")

    def _init_plots(self):
        """Create initial plot elements that will be updated in-place."""
        # Field heatmap (initial dummy data)
        dummy_field = np.zeros((D, P))
        self._field_img = self.ax_field.imshow(
            dummy_field, aspect='auto', cmap='inferno', interpolation='bilinear'
        )
        self.ax_field.set_title("FHIDS Field")
        self.ax_field.set_xlabel("Pins")
        self.ax_field.set_ylabel("Dimensions")
        self._field_cbar = self.fig.colorbar(self._field_img, ax=self.ax_field)

        # Energy line
        self._energy_line, = self.ax_energy.plot([], [], color='lime', linewidth=2)
        self.ax_energy.set_title("Energy Evolution (last 100)")
        self.ax_energy.set_xlabel("Step")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.grid(True, alpha=0.3)

        # Entropy line
        self._entropy_line, = self.ax_entropy.plot([], [], color='magenta', linewidth=2)
        self.ax_entropy.set_title("Entropy Evolution (last 100)")
        self.ax_entropy.set_xlabel("Sample")
        self.ax_entropy.set_ylabel("Entropy")
        self.ax_entropy.grid(True, alpha=0.3)

    def run(self):
        """Start visualization loop."""
        while True:
            try:
                # Get field (use magnitude for complex mode)
                field = self.engine.diag.reshape(P, D)
                display_field = np.abs(field) if np.iscomplexobj(field) else field

                # Plot 1: Update field heatmap data (no recreate)
                self._field_img.set_data(display_field.T)
                self._field_img.set_clim(vmin=display_field.min(),
                                         vmax=display_field.max())

                # Plot 2: Pin distribution (must redraw - histogram can't update in-place)
                self.ax_pins.clear()
                self.ax_pins.hist(self.engine.pins, bins=20, color='cyan', alpha=0.7, edgecolor='white')
                self.ax_pins.axvline(self.engine.target_pin, color='red', linestyle='--', label='Target')
                self.ax_pins.set_title("Pin Frequencies Distribution")
                self.ax_pins.set_xlabel("Frequency (0-100)")
                self.ax_pins.set_ylabel("Count")
                self.ax_pins.legend()

                # Get metrics
                metrics = self.engine.export_semantic()

                # Update history
                self.energy_history.append(metrics['energy'])
                self.entropy_history.append(metrics['entropy'])
                self.step_history.append(self.engine.step)

                # Plot 3: Update energy line data
                recent_steps = list(self.step_history)[-100:]
                recent_energy = list(self.energy_history)[-100:]
                self._energy_line.set_data(recent_steps, recent_energy)
                self.ax_energy.relim()
                self.ax_energy.autoscale_view()

                # Plot 4: Update entropy line data
                recent_entropy = list(self.entropy_history)[-100:]
                self._entropy_line.set_data(range(len(recent_entropy)), recent_entropy)
                self.ax_entropy.relim()
                self.ax_entropy.autoscale_view()

                # Refresh
                plt.pause(self.update_interval)

            except Exception as e:
                print(f"[VISUALIZER] Error: {e}")
                time.sleep(1)


def launch_visualizer(engine):
    """Start visualizer in background thread."""
    viz = THCVisualizer(engine)
    thread = threading.Thread(target=viz.run, daemon=True)
    thread.start()
    return viz


# Backward compatibility alias
HFBPVisualizer = THCVisualizer
