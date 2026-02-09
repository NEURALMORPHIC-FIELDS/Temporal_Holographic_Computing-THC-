#!/usr/bin/env python3
"""
THC UI Control Panel (Tkinter) — Temporal Holographic Computation
Live parameter control + monitoring
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from config import P, D, LEVELS


class THCControlPanel:
    def __init__(self, engine):
        """Initialize control panel."""
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("THC Control Panel")
        self.root.geometry("500x600")

        # Status
        self.status_var = tk.StringVar(value="Initializing...")

        # Build UI
        self._build_ui()

        # Schedule periodic updates on the main thread (thread-safe)
        self._schedule_monitor()

        print("[UI] Control panel initialized")

    def _build_ui(self):
        """Build Tkinter interface."""

        # ===== TITLE =====
        title = tk.Label(self.root, text="THC GPU Engine Control",
                         font=("Arial", 14, "bold"), fg="#00FF00", bg="black")
        title.pack(fill='x', padx=5, pady=5)

        # ===== PARAMETERS =====
        params_frame = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)

        # Target Pin
        tk.Label(params_frame, text="Target Pin").grid(row=0, column=0, sticky='w')
        self.scale_target = tk.Scale(params_frame, from_=60, to=90, resolution=1,
                                     orient='horizontal',
                                     command=self._set_target)
        self.scale_target.set(int(self.engine.target_pin))
        self.scale_target.grid(row=0, column=1, sticky='ew')

        # Tune Rate
        tk.Label(params_frame, text="Tune Rate").grid(row=1, column=0, sticky='w')
        self.scale_rate = tk.Scale(params_frame, from_=0.01, to=0.1, resolution=0.005,
                                   orient='horizontal',
                                   command=self._set_rate, length=200)
        self.scale_rate.set(self.engine.tune_rate)
        self.scale_rate.grid(row=1, column=1, sticky='ew')

        # Coupling Alpha
        tk.Label(params_frame, text="Coupling Alpha").grid(row=2, column=0, sticky='w')
        self.scale_alpha = tk.Scale(params_frame, from_=0.0, to=0.2, resolution=0.01,
                                    orient='horizontal',
                                    command=self._set_alpha)
        self.scale_alpha.set(self.engine.alpha)
        self.scale_alpha.grid(row=2, column=1, sticky='ew')

        params_frame.columnconfigure(1, weight=1)

        # ===== METRICS =====
        metrics_frame = ttk.LabelFrame(self.root, text="Metrics", padding=10)
        metrics_frame.pack(fill='x', padx=5, pady=5)

        self.label_step = tk.Label(metrics_frame, text="Step: 0", font=("Courier", 10))
        self.label_step.pack(anchor='w')

        self.label_energy = tk.Label(metrics_frame, text="Energy: 0.0", font=("Courier", 10))
        self.label_energy.pack(anchor='w')

        self.label_variance = tk.Label(metrics_frame, text="Variance: 0.0", font=("Courier", 10))
        self.label_variance.pack(anchor='w')

        self.label_entropy = tk.Label(metrics_frame, text="Entropy: 0.0", font=("Courier", 10))
        self.label_entropy.pack(anchor='w')

        self.label_stable = tk.Label(metrics_frame, text="Stable Ctr: 0", font=("Courier", 10))
        self.label_stable.pack(anchor='w')

        # ===== STATUS =====
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)

        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                     font=("Courier", 10), wraplength=400, justify='left')
        self.status_label.pack(anchor='w')

        # ===== CONTROLS =====
        controls_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        controls_frame.pack(fill='x', padx=5, pady=5)

        btn_checkpoint = tk.Button(controls_frame, text="Save Checkpoint",
                                   command=self._save_checkpoint)
        btn_checkpoint.pack(side='left', padx=2)

        btn_inject = tk.Button(controls_frame, text="Inject Stimulus",
                               command=self._inject_stimulus)
        btn_inject.pack(side='left', padx=2)

        btn_export = tk.Button(controls_frame, text="Export Semantic",
                               command=self._export_semantic)
        btn_export.pack(side='left', padx=2)

        # ===== TEXT INPUT =====
        input_frame = ttk.LabelFrame(self.root, text="Text Stimulus", padding=10)
        input_frame.pack(fill='x', padx=5, pady=5)

        self.text_input = tk.Entry(input_frame, width=50)
        self.text_input.pack(fill='x', padx=2)
        self.text_input.insert(0, "Type stimulus text here...")

    def _set_target(self, value):
        self.engine.target_pin = float(value)

    def _set_rate(self, value):
        self.engine.tune_rate = float(value)

    def _set_alpha(self, value):
        self.engine.alpha = float(value)

    def _save_checkpoint(self):
        self.engine.save_checkpoint(self.engine.step)
        self.status_var.set(f"Checkpoint saved at step {self.engine.step}")

    def _inject_stimulus(self):
        text = self.text_input.get()
        if text and text != "Type stimulus text here...":
            self.engine.inject_text_stimulus(text, gain=0.01)
            self.status_var.set(f"Injected: {text[:30]}...")

    def _export_semantic(self):
        self.engine.write_semantic("semantic_export.json")
        self.status_var.set(f"Exported semantic to semantic_export.json")

    def _schedule_monitor(self):
        """Schedule periodic metric updates on the Tkinter main thread."""
        self._update_metrics()

    def _update_metrics(self):
        """Update metric labels. Runs on main thread via root.after()."""
        try:
            metrics = self.engine.export_semantic()

            self.label_step.config(text=f"Step: {self.engine.step}")
            self.label_energy.config(text=f"Energy: {metrics['energy']:.6f}")
            self.label_variance.config(text=f"Variance: {metrics['variance']:.6f}")
            self.label_entropy.config(text=f"Entropy: {metrics['entropy']:.6f}")
            self.label_stable.config(text=f"Stable Ctr: {self.engine.stable_ctr}")
        except Exception:
            pass  # Engine may not be ready yet

        # Reschedule every 500ms
        self.root.after(500, self._update_metrics)

    def run(self):
        """Start UI loop."""
        self.root.mainloop()


# Backward compatibility alias
HFBPControlPanel = THCControlPanel
