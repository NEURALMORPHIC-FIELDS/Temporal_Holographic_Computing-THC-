#!/usr/bin/env python3
"""
THC GPU Engine v3.0 — Temporal Holographic Computation
AMD RX 6700 XT | OpenCL 2.x | Windows

Complex Phase Architecture (THC-Phi)
  - Complex-valued state psi in C (float2 / complex64)
  - Phase-preserving force: tanh(|z|) * z/|z|
  - Leapfrog integration (symplectic)
  - Delay feedback with complex delay ring
  - Pin-field coupling (magnitude-based)
  - FHIDS Z=64 holographic storage
  - Stochastic driving (real noise -> Re(psi))

Set COMPLEX_MODE = False in config.py for v2.x backward compatibility.
"""

import pyopencl as cl
import numpy as np
import time
import os
import json
from collections import deque
from pathlib import Path

from config import (
    P, D, LEVELS, Z, DT, TARGET_PIN, TUNE_RATE, ALPHA,
    TARGET_E, STABLE_STEPS, SLEEP_MS, CHECKPOINT_DIR,
    AUTO_CHECKPOINT, AUTONOMY_ENABLED, VERBOSE,
    TAU, DELAY_STRENGTH, VELOCITY_DAMPING,
    PIN_COUPLING_EPSILON, PIN_COUPLING_RADIUS, NOISE_AMP,
    COMPLEX_MODE
)

# Pre-computed constants
_INT32_D = np.int32(D)
_INT32_P = np.int32(P)
_INT32_Z = np.int32(Z)
_INT32_R = np.int32(PIN_COUPLING_RADIUS)
_FLOAT32_DT = np.float32(DT)
_FLOAT32_DELAY = np.float32(DELAY_STRENGTH)
_FLOAT32_DAMPING = np.float32(VELOCITY_DAMPING)
_GLOBAL_SIZE_PD = (P * D,)
_GLOBAL_SIZE_P = (P,)
_GLOBAL_SIZE_2D = (P, D)
_LOCAL_SIZE_PD = (min(64, P * D),)
_LOCAL_SIZE_P = (min(64, P),)
_LOCAL_SIZE_2D = (min(64, P), 1)

_MAX_HISTORY = 10000


class THCEngine:
    def __init__(self):
        """Initialize THC GPU engine v3.0."""

        # v3.0: complex-valued state dtype
        self.complex_mode = COMPLEX_MODE
        self.dtype = np.complex64 if COMPLEX_MODE else np.float32
        self.element_size = 8 if COMPLEX_MODE else 4  # bytes per element

        # OpenCL setup
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        print("[THC] Context:", self.ctx.devices[0].name)
        print(f"[THC] Platform: {self.ctx.devices[0].platform.name}")
        print(f"[THC] Mode: {'COMPLEX (v3.0 THC-Phi)' if COMPLEX_MODE else 'REAL (v2.x)'}")

        # Load kernels
        kernel_path = Path(__file__).parent / "kernels.cl"
        with open(kernel_path) as f:
            kernel_src = f.read()

        build_options = "-D COMPLEX_MODE" if COMPLEX_MODE else ""
        self.program = cl.Program(self.ctx, kernel_src).build(options=build_options)

        # Cache all kernels
        self.kernel_dfpm = cl.Kernel(self.program, 'dfpm')
        self.kernel_fpis = cl.Kernel(self.program, 'fpis')
        self.kernel_fda = cl.Kernel(self.program, 'fda')
        self.kernel_fhids_store = cl.Kernel(self.program, 'fhids_store')
        self.kernel_fhids_diag = cl.Kernel(self.program, 'fhids_diag')
        self.kernel_couple = cl.Kernel(self.program, 'couple_levels')
        self.kernel_update = cl.Kernel(self.program, 'update_state')
        self.kernel_evolve = cl.Kernel(self.program, 'evolve_level')
        # v2.0 kernels
        self.kernel_leapfrog = cl.Kernel(self.program, 'evolve_leapfrog')
        self.kernel_pin_coupling = cl.Kernel(self.program, 'pin_field_coupling')
        self.kernel_copy = cl.Kernel(self.program, 'copy_buffer')
        self.kernel_add_noise = cl.Kernel(self.program, 'add_noise')
        # v2.1: holographic binding (Direction 2: HRR/VSA)
        self.kernel_bind = cl.Kernel(self.program, 'holographic_bind')

        print(f"[THC] Kernels compiled & cached OK (v3.0: 13 kernels, "
              f"{'complex' if COMPLEX_MODE else 'real'} mode)")

        # Host memory
        if COMPLEX_MODE:
            # v3.0: complex state — real part N(0,0.5), imag part N(0,0.1)
            self.psi_levels = [
                (np.random.randn(P, D).astype(np.float32) * 0.5
                 + 1j * np.random.randn(P, D).astype(np.float32) * 0.1
                 ).astype(np.complex64)
                for _ in range(LEVELS)
            ]
        else:
            self.psi_levels = [np.random.randn(P, D).astype(np.float32) * 0.5
                               for _ in range(LEVELS)]
        self.psi_prev_levels = [psi.copy() for psi in self.psi_levels]
        # Supercritical pin initialization (always real)
        self.pins = np.random.uniform(105, 160, size=P).astype(np.float32)
        self.holo = np.zeros((Z, D), dtype=self.dtype)
        self.diag = np.zeros(P * D, dtype=self.dtype)
        self.dpsi = np.zeros(P * D, dtype=self.dtype)
        # Velocity field (one per level, same dtype as psi)
        self.vel_levels = [np.zeros((P, D), dtype=self.dtype) for _ in range(LEVELS)]

        # GPU buffers
        mf = cl.mem_flags

        self.psi_bufs = [
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=psi)
            for psi in self.psi_levels
        ]
        self.psi_prev_bufs = [
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=psi)
            for psi in self.psi_prev_levels
        ]
        self.pins_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                  hostbuf=self.pins)
        self.holo_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                                  hostbuf=self.holo)
        self.diag_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.diag.nbytes)
        self.dpsi_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.dpsi.nbytes)

        # Velocity buffers
        self.vel_bufs = [
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
            for vel in self.vel_levels
        ]

        # Delay ring buffer [TAU slots][LEVELS]
        psi_byte_size = P * D * self.element_size
        self.delay_ring = []
        for t in range(TAU):
            level_bufs = []
            for l in range(LEVELS):
                buf = cl.Buffer(self.ctx, mf.READ_WRITE, psi_byte_size)
                cl.enqueue_copy(self.queue, buf, self.psi_levels[l])
                level_bufs.append(buf)
            self.delay_ring.append(level_bufs)

        # Noise buffer (always real — injected into Re(psi) in complex mode)
        self.noise_host = np.zeros(P * D, dtype=np.float32)
        self.noise_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.noise_host.nbytes)

        self.queue.finish()

        print(f"[THC] GPU buffers allocated "
              f"(P={P}, D={D}, Z={Z}, LEVELS={LEVELS}, TAU={TAU}, "
              f"element={self.element_size}B)")

        # State
        self.step = 0
        self.running = True
        self.stable_ctr = 0
        self.delta_E_history = deque(maxlen=_MAX_HISTORY)

        # Control parameters (mutable)
        self.target_pin = TARGET_PIN
        self.tune_rate = TUNE_RATE
        self.alpha = ALPHA

        # Network bridge reference (set by main.py)
        self.network = None

        Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

        version = "v3.0 THC-Phi (complex)" if COMPLEX_MODE else "v2.x (real)"
        print(f"[THC] {version} Engine initialized")
        print(f"[THC]   Dynamics: Leapfrog + delay(tau={TAU-1}) + "
              f"pin-coupling(eps={PIN_COUPLING_EPSILON}, R={PIN_COUPLING_RADIUS})")
        print(f"[THC]   Pins: supercritical [105,160], target={TARGET_PIN}")
        print(f"[THC]   FHIDS: Z={Z} ({P*D//(Z*D)}:1 compression)")

    def step_engine(self):
        """Single evolution step: Leapfrog + delay + pin coupling + noise."""

        f32_target = np.float32(self.target_pin)
        f32_rate = np.float32(self.tune_rate)
        f32_alpha = np.float32(self.alpha)
        f32_epsilon = np.float32(PIN_COUPLING_EPSILON)

        # Delay buffer indices
        write_slot = self.step % TAU
        read_slot = (self.step + 1) % TAU  # oldest = effective delay TAU-1

        # === Per-level Leapfrog evolution ===
        for l in range(LEVELS):
            # Save current psi to delay ring (GPU-side copy)
            self.kernel_copy(self.queue, _GLOBAL_SIZE_PD, _LOCAL_SIZE_PD,
                             self.delay_ring[write_slot][l], self.psi_bufs[l])

            # Leapfrog with delay feedback
            self.kernel_leapfrog(
                self.queue, _GLOBAL_SIZE_2D, _LOCAL_SIZE_2D,
                self.psi_bufs[l],
                self.vel_bufs[l],
                self.delay_ring[read_slot][l],
                self.pins_buf,
                self.dpsi_buf,
                _INT32_D, _FLOAT32_DT,
                _FLOAT32_DELAY, _FLOAT32_DAMPING
            )

        # === Stochastic driving ===
        if NOISE_AMP > 0:
            self.noise_host[:] = (np.random.randn(P * D) * NOISE_AMP).astype(np.float32)
            cl.enqueue_copy(self.queue, self.noise_buf, self.noise_host)
            self.kernel_add_noise(self.queue, _GLOBAL_SIZE_PD, _LOCAL_SIZE_PD,
                                  self.psi_bufs[0], self.noise_buf)

        # === Pin-Field Coupling (FIX 1) ===
        self.kernel_pin_coupling(
            self.queue, _GLOBAL_SIZE_P, _LOCAL_SIZE_P,
            self.pins_buf, self.psi_bufs[0],
            _INT32_D, _INT32_P, _INT32_R,
            f32_epsilon, f32_target, f32_rate
        )

        # === Multi-scale coupling ===
        for l in range(LEVELS - 1):
            self.kernel_couple(self.queue, _GLOBAL_SIZE_PD, _LOCAL_SIZE_PD,
                               self.psi_bufs[l], self.psi_bufs[l + 1], f32_alpha)

        # === FHIDS (Z=64, 4:1 compression) ===
        self.kernel_fhids_store(self.queue, _GLOBAL_SIZE_PD, _LOCAL_SIZE_PD,
                                self.holo_buf, self.psi_bufs[0],
                                _INT32_Z, _INT32_D)
        self.kernel_fhids_diag(self.queue, _GLOBAL_SIZE_PD, _LOCAL_SIZE_PD,
                               self.holo_buf, self.diag_buf,
                               _INT32_Z, _INT32_D)

        self.queue.finish()

        # Readback metrics
        cl.enqueue_copy(self.queue, self.dpsi, self.dpsi_buf)
        cl.enqueue_copy(self.queue, self.pins, self.pins_buf)
        self.queue.finish()

        delta_E = float(np.mean(np.abs(self.dpsi)))
        pin_var = float(np.var(self.pins))

        self.step += 1
        self.delta_E_history.append(delta_E)

        return delta_E, pin_var

    def _sync_host_state(self):
        """Full readback of GPU state to host arrays."""
        for l in range(LEVELS):
            cl.enqueue_copy(self.queue, self.psi_levels[l], self.psi_bufs[l])
            cl.enqueue_copy(self.queue, self.vel_levels[l], self.vel_bufs[l])
        cl.enqueue_copy(self.queue, self.diag, self.diag_buf)
        cl.enqueue_copy(self.queue, self.holo, self.holo_buf)
        cl.enqueue_copy(self.queue, self.pins, self.pins_buf)
        self.queue.finish()

    def regulate(self, delta_E):
        """Attractor control: stabilize if E drifts."""
        if delta_E > TARGET_E:
            self.tune_rate *= 0.95
        else:
            self.tune_rate *= 1.01
        self.tune_rate = np.clip(self.tune_rate, 0.001, 0.1)

    def autonomy_step(self, delta_E):
        """Self-governance: checkpoint when stable."""
        if delta_E < TARGET_E:
            self.stable_ctr += 1
        else:
            self.stable_ctr = 0

        if self.stable_ctr == STABLE_STEPS and AUTO_CHECKPOINT:
            self._sync_host_state()
            self.save_checkpoint(self.step)
            self.stable_ctr = 0
            if VERBOSE:
                print(f"[AUTONOMY] Checkpoint saved at step {self.step}")

    def save_checkpoint(self, step):
        """Save system state to NPZ."""
        self._sync_host_state()

        data = {
            "step": step,
            "pins": self.pins,
            "holo": self.holo,
            "diag": self.diag,
            "tune_rate": self.tune_rate,
            "delta_E_history": np.array(self.delta_E_history)
        }
        for l in range(LEVELS):
            data[f"psi_{l}"] = self.psi_levels[l]
            data[f"vel_{l}"] = self.vel_levels[l]

        path = Path(CHECKPOINT_DIR) / f"checkpoint_{step:08d}.npz"
        np.savez(path, **data)

        if VERBOSE:
            print(f"[CHECKPOINT] Saved: {path}")

    def _convert_dtype(self, arr):
        """Convert array to current engine dtype (handles cross-mode checkpoint loading)."""
        if arr.dtype == self.dtype:
            return arr
        if COMPLEX_MODE and np.issubdtype(arr.dtype, np.floating):
            # Upgrading real -> complex: real part = loaded, imag = 0
            return arr.astype(np.complex64)
        if not COMPLEX_MODE and np.issubdtype(arr.dtype, np.complexfloating):
            # Downgrading complex -> real: take real part
            return arr.real.astype(np.float32)
        return arr.astype(self.dtype)

    def load_checkpoint(self, path):
        """Restore system state from checkpoint (handles cross-mode dtype)."""
        data = np.load(path, allow_pickle=True)

        self.pins[:] = data["pins"].astype(np.float32)
        self.tune_rate = float(data["tune_rate"])
        self.step = int(data["step"])

        if "holo" in data:
            loaded_holo = self._convert_dtype(data["holo"])
            if loaded_holo.shape == self.holo.shape:
                self.holo[:] = loaded_holo
        if "diag" in data:
            loaded_diag = self._convert_dtype(data["diag"])
            if loaded_diag.shape == self.diag.shape:
                self.diag[:] = loaded_diag

        for l in range(LEVELS):
            self.psi_levels[l][:] = self._convert_dtype(data[f"psi_{l}"])
            self.psi_prev_levels[l][:] = self.psi_levels[l]
            if f"vel_{l}" in data:
                self.vel_levels[l][:] = self._convert_dtype(data[f"vel_{l}"])
            else:
                self.vel_levels[l][:] = 0

        cl.enqueue_copy(self.queue, self.pins_buf, self.pins)
        for l in range(LEVELS):
            cl.enqueue_copy(self.queue, self.psi_bufs[l], self.psi_levels[l])
            cl.enqueue_copy(self.queue, self.vel_bufs[l], self.vel_levels[l])
            for t in range(TAU):
                cl.enqueue_copy(self.queue, self.delay_ring[t][l], self.psi_levels[l])

        self.queue.finish()

        if VERBOSE:
            print(f"[CHECKPOINT] Loaded: {path} (step {self.step})")

    def inject_text_stimulus(self, text, gain=0.05):
        """Inject text-based stimulus (hash -> perturbation into Re(psi))."""
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        v = (v / 255.0 - 0.5) * gain

        host_psi = np.empty_like(self.psi_levels[0])
        cl.enqueue_copy(self.queue, host_psi, self.psi_bufs[0])
        self.queue.finish()

        n = min(P, len(v))
        if COMPLEX_MODE:
            # Inject into real part only — phase evolves from dynamics
            host_psi[:n, 0] += v[:n].astype(np.float32)
        else:
            host_psi[:n, 0] += v[:n]
        cl.enqueue_copy(self.queue, self.psi_bufs[0], host_psi)
        self.queue.finish()

        self.psi_levels[0] = host_psi

        if VERBOSE:
            print(f"[STIMULUS] Text: '{text[:30]}...' injected")

    def export_semantic(self):
        """Export semantic snapshot from FHIDS."""
        cl.enqueue_copy(self.queue, self.diag, self.diag_buf)
        self.queue.finish()

        field = self.diag.reshape(P, D)
        abs_field = np.abs(field)  # works for both real and complex

        return {
            "energy": float(np.mean(abs_field)),
            "variance": float(np.mean(abs_field ** 2) - np.mean(abs_field) ** 2),
            "centroid": np.abs(field).mean(axis=0).tolist(),
            "entropy": float(-np.sum(abs_field * np.log(abs_field + 1e-8))),
            "step": self.step
        }

    def write_semantic(self, path="semantic.json"):
        """Write semantic snapshot to JSON."""
        with open(path, "w") as f:
            json.dump(self.export_semantic(), f, indent=2)

    def run_loop(self, max_steps=None):
        """Main evolution loop."""
        mode = "v3.0 complex" if COMPLEX_MODE else "v2.x real"
        print(f"[ENGINE] Starting evolution loop ({mode} dynamics)...")

        sleep_sec = SLEEP_MS / 1000.0

        try:
            while self.running and (max_steps is None or self.step < max_steps):
                delta_E, pin_var = self.step_engine()

                self.regulate(delta_E)

                if AUTONOMY_ENABLED:
                    self.autonomy_step(delta_E)

                if self.network is not None:
                    self.network.poll()

                if self.step % 100 == 0:
                    metrics = self.export_semantic()
                    print(f"[STEP {self.step:06d}] dE={delta_E:.6f} | "
                          f"Var(Pins)={pin_var:.4f} | "
                          f"Energy={metrics['energy']:.4f} | "
                          f"Entropy={metrics['entropy']:.2f}")

                time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("\n[ENGINE] Interrupted by user")
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        self.queue.finish()
        print("[ENGINE] Shutdown complete")


# Backward compatibility alias
HFBPEngine = THCEngine

# ==================== MAIN ====================
if __name__ == "__main__":
    engine = THCEngine()

    try:
        engine.run_loop(max_steps=10000)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.stop()
