![THC Logo](logo.png)

# THC Technical Specification v3.0

## Executive Summary

THC (Temporal Holographic Computation) is a **GPU-accelerated non-symbolic computation engine** designed for AMD hardware, specifically validated on RX 6700 XT.

Unlike traditional AI (neural networks, transformers), THC operates as a **dynamical system** with:
- Continuous state evolution (Ψ)
- Multi-scale fractal structure
- Holographic memory (FHIDS)
- Feedback-driven stability
- Autonomous persistence

> **v3.0: Complex Phase Architecture** - This version introduces expanded holographic planes (Z=64), symplectic integration (leapfrog), complex phase binding, and 13 GPU kernels for full temporal-holographic computation.

This document specifies the complete architecture, algorithms, and integration points.

---

## 1. System Architecture

### 1.1 Hardware Target
- **GPU**: AMD Radeon RX 6700 XT (RDNA2, 40 CUs, 12GB VRAM)
- **Compute API**: OpenCL 2.x (via ROCm)
- **Host**: Windows 10/11 (user-space process)
- **Memory**: ~100MB GPU allocation (safe margin)

### 1.2 Software Stack

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│  ┌─────────────────────────────────┤
│  │ main.py (orchestration)         │
│  ├─────────────────────────────────┤
│  │ UI (Tkinter) | Network (TCP)    │
│  │ Visualizer (matplotlib)*         │
│  └─────────────────────────────────┘
│   Engine Layer                      │
│  ┌─────────────────────────────────┤
│  │ THCEngine (Python)             │
│  │ - State management              │
│  │ - Checkpoint I/O                │
│  │ - Metrics computation           │
│  │ - Autonomy control              │
│  └─────────────────────────────────┘
│   GPU Compute Layer                 │
│  ┌─────────────────────────────────┤
│  │ OpenCL Runtime + PyOpenCL       │
│  │ 13 Kernels: DFPM, FPIS, FDAΨ,  │
│  │   FHIDS, coupling, leapfrog,    │
│  │   pin_coupling, copy_buffer,    │
│  │   add_noise, holographic_bind   │
│  │ Buffers: Ψ[L], pins, holo, diag │
│  └─────────────────────────────────┘
└─────────────────────────────────────┘
        AMD Adrenalin Driver
        AMD GPU (RDNA2 ISA)
```

*Optional: visualizer.py not required for core operation.

---

## 2. Core Model

### 2.1 Mathematical Definition

**State vector Ψ**:
```
Ψ ∈ ℝ^(P × D × L)

P   = 256 pins (frequency identity units)
D   = 16 dimensions (aspect of each pin)
L   = 3 levels (multi-scale hierarchy)
```

Each Ψ[p, d, l] is a scalar in [-∞, +∞].

**Evolution equation**:
```
Ψ^(l)_(t+1) = P_l(  A_l(Ψ^(l)_t) + B_l(Ψ^(l)_prev) + ξ_t  )

where:
  A_l = tanh(Ψ^(l)_t × pin_frequency/100) - local nonlinear transform
  B_l = feedback coupling (level-dependent)
  P_l = projection (normalization via tanh)
  ξ_t = stochastic perturbation (controlled noise)
```

**Frequency identity update**:
```
pins_t+1[p] = pins_t[p] + r × (target - pins_t[p])

r = tune_rate ∈ [0.01, 0.1]  (adaptive)
target = 130.0               (attractor)
```

**Holographic storage (FHIDS)**:
```
holo[z, d] = Ψ^(0)[z % Z, d % D]

Encodes temporal state in multi-plane memory.
Diagonal read (K5) reconstructs via interference.
```

### 2.2 Stability & Control

**Energy metric**:
```
E_t = mean(|Ψ_(t+1) - Ψ_t|)

Indicates convergence speed.
Target: E < 0.002 (stable regime)
```

**Attractor mechanism**:
```
if E_t > TARGET_E:
    tune_rate *= 0.95    (dampen)
else:
    tune_rate *= 1.01    (explore)

Bounds: tune_rate ∈ [0.01, 0.1]
```

**Autonomy criterion**:
```
if E_t < TARGET_E continuously for STABLE_STEPS:
    save_checkpoint()
    reset_stable_counter()

Ensures self-directed persistence without external agents.
```

---

## 3. GPU Kernels

### 3.1 Kernel Summary

| Kernel | Input | Output | Op Count | Notes |
|--------|-------|--------|----------|-------|
| DFPM   | Ψ, pins | Ψ | P×D×tanh | Bulk fractal |
| FPIS   | pins, target | pins | P additions | Frequency update |
| FDAΨ   | Ψ, Ψ_prev | Ψ | P×D divs | Finite difference |
| FHIDS_store | Ψ, holo | holo | Z×D writes | Holographic encode |
| FHIDS_diag | holo | diag | Z×D reads | Holographic decode |
| couple_levels | Ψ^l, Ψ^(l+1) | Ψ^l | (L-1)×P×D ops | Recursive damping |
| K9: leapfrog | Ψ, Ψ_prev, dt | Ψ | P×D ops | Symplectic integrator |
| K10: pin_coupling | pins, Ψ | pins | P ops | Phase-frequency bind |
| K11: copy_buffer | src | dst | N copies | State duplication |
| K12: add_noise | Ψ, seed | Ψ | P×D ops | Stochastic perturbation |
| K13: holographic_bind | holo, Ψ | holo | Z×D ops | Complex phase binding |

**Total throughput**: ~10 million ops/iteration on RX 6700 XT (~1-2ms per step).

### 3.2 Kernel Grammar (OpenCL C)

All kernels:
- Use `__global` memory (VRAM)
- No `__local` shared memory (for simplicity; can optimize)
- Aligned access patterns (coalesced reads/writes)
- Work-group size: global_size = (P,) or (P×D,)
- Barriers: implicit at kernel boundaries

Example (DFPM):
```c
__kernel void dfpm(
    __global float* psi,      // P×D array, row-major
    __global float* pins,     // P array
    int D
) {
    int i = get_global_id(0); // Thread index ∈ [0, P)
    float p = pins[i];
    int base = i * D;
    
    for (int d = 0; d < D; d++) {
        float x = psi[base + d];
        psi[base + d] = tanh(x * (p / 100.0f));
    }
}
```

No barrier needed (thread-local loop, no inter-pin sync).

---

## 4. Memory Layout

### 4.1 Host Memory (numpy arrays)

```python
psi_levels[l]        shape (P, D) = (256, 16)      float32
psi_prev_levels[l]   shape (P, D) = (256, 16)      float32
pins                 shape (P,)   = (256,)         float32
holo                 shape (Z, D) = (64, 16)       float32
diag                 shape (P*D,) = (4096,)        float32
```

Row-major (C-contiguous) for efficient GPU upload.

### 4.2 GPU Memory (OpenCL buffers)

```
Total: P×D×4 + P×D×4 + P×4 + Z×D×4 + P×D×4
     = 256×16×4 + ... 
     ≈ 100 KB (negligible; 12GB available)
```

**Buffer lifecycle**:
1. `enqueue_copy(host -> GPU)` - Upload state
2. Kernel execution (in-place or ping-pong)
3. `enqueue_copy(GPU -> host)` - Readback metrics
4. Repeat

---

## 5. Execution Flow

### 5.1 Main Loop Pseudocode

```python
while running:
    # Poll external stimuli
    network_bridge.poll()
    
    # GPU evolution (all levels)
    for l in range(LEVELS):
        queue.enqueue(dfpm, psi_levels[l], pins)
        queue.enqueue(fpis, pins, target, tune_rate)
        queue.enqueue(fda, psi_levels[l], psi_prev_levels[l], dt)
    
    # Inter-level coupling
    for l in range(LEVELS-1):
        queue.enqueue(couple_levels, psi_levels[l], psi_levels[l+1], alpha)
    
    # Holographic storage
    queue.enqueue(fhids_store, holo, psi_levels[0], Z, D)
    queue.enqueue(fhids_diag, holo, diag, Z, D)
    
    # Synchronize
    queue.finish()
    
    # Readback metrics
    enqueue_copy(psi_levels[0], psi_buffers[0])
    enqueue_copy(pins, pins_buf)
    enqueue_copy(diag, diag_buf)
    
    # Host-side control
    delta_E = compute_energy()
    regulate(delta_E)
    
    # Autonomy
    if stable_ctr > STABLE_STEPS:
        save_checkpoint()
    
    # UI update (non-blocking)
    metrics = export_semantic()
    
    # Throttle (OS fairness)
    sleep(1 ms)
```

**Timing per iteration**:
- GPU kernel execution: ~1-2 ms
- Host readback: <1 ms
- Host control: <1 ms
- Sleep: ~1 ms
- **Total latency**: 2-5 ms (500 Hz effective)

### 5.2 Startup Sequence

```
1. Parse config
2. Init OpenCL context (auto-detect AMD GPU)
3. Compile 13 kernels from kernels.cl
4. Allocate GPU buffers
5. Random initialize state (Ψ, pins, holo)
6. (Optional) load checkpoint
7. Create checkpoint directory
8. Spawn engine thread (background)
9. (Optional) spawn network server
10. (Optional) launch UI
11. Start main loop
```

---

## 6. Persistence (Checkpoints)

### 6.1 Checkpoint Format

File: `checkpoint_XXXXXXXX.npz` (numpy compressed)

Contents:
```python
{
    "step": int,                  # Global iteration count
    "pins": ndarray (P,),        # Current frequency state
    "psi_0": ndarray (P, D),     # Level 0 field
    "psi_1": ndarray (P, D),     # Level 1 field
    "psi_2": ndarray (P, D),     # Level 2 field
    "holo": ndarray (Z, D),      # Holographic memory
    "diag": ndarray (P*D,),      # Diagonal projection
    "tune_rate": float,           # Control parameter
    "delta_E_history": ndarray    # Energy trajectory
}
```

Size: ~10 MB per checkpoint.

### 6.2 Autonomy Trigger

```
if E_t < TARGET_E for STABLE_STEPS consecutive iterations:
    save_checkpoint(step)
    reset_stable_counter()
```

This ensures natural "resting points" in phase space.

---

## 7. External Interfaces

### 7.1 Network Bridge (TCP)

**Protocol**: Plain text over TCP/IP

**Server**: 127.0.0.1:7777 (configurable)

**Client sends**: Any string, hash-encoded to perturbation

**Example**:
```bash
echo "concept: emergence" | nc localhost 7777
```

Internally:
```python
h = SHA256("concept: emergence".encode())
v = (unpack_bytes(h) / 255.0 - 0.5) * 0.005  # gain
Ψ[0, :, 0] += v                               # inject
```

Stimulus applied only when system is stable (safe stochastic coupling).

### 7.2 Control Panel (Tkinter)

**Live parameters** (sliders):
- Target Pin: [60, 90]
- Tune Rate: [0.01, 0.1]
- Coupling Alpha: [0.0, 0.2]

**Buttons**:
- Save Checkpoint
- Inject Text Stimulus
- Export Semantic JSON

**Readouts**:
- Step count
- Energy
- Variance
- Entropy
- Stability counter

**Threading**: UI runs on main thread, engine in background.

### 7.3 JSON Semantic Export

```json
{
  "step": 10052,
  "energy": 0.001234,
  "variance": 0.05678,
  "entropy": 3.456,
  "centroid": [0.1, 0.2, ..., 0.16]
}
```

Used for external analysis or LLM conditioning.

---

## 8. Configuration Parameters

### 8.1 System (immutable at runtime)

```python
P = 256           # Pins
D = 16            # Dimensions
LEVELS = 3        # Fractal depth
Z = 64             # Holographic planes
```

Rationale:
- P = 256: wavefront-aligned (8×32), optimal for 40 CUs on RX 6700 XT
- D = 16: balance between diversity and memory
- LEVELS = 3: fast (3 coupling layers), sufficient emergent structure
- Z = 64: hologram resolution, expanded phase capacity

### 8.2 Dynamics (mutable at runtime)

```python
TARGET_PIN = 130.0        # Attractor (frequency)
TUNE_RATE = 0.05         # Adaptation rate [0.01, 0.1]
ALPHA = 0.05             # Inter-level coupling
DT = 0.1                 # Time step
```

### 8.3 Control (mutable at runtime)

```python
TARGET_E = 0.002         # Stability threshold
STABLE_STEPS = 200       # Autonomy window
```

### 8.4 Runtime Behavior

```python
SLEEP_MS = 1             # Throttle (OS fairness)
AUTO_CHECKPOINT = True   # Enable autonomy
AUTONOMY_ENABLED = True  # Attractor regulation
```

---

## 9. Performance & Resource Usage

### 9.1 Bandwidth

**GPU memory transfer per iteration**:
```
Upload:   P×D×4 + P×4                    ≈ 16 KB
Compute:  (P×D) kernel ops on VRAM       in-place
Readback: P×D×4 + P×4 + P*D×4           ≈ 32 KB
Total:    ~50 KB / 2-5 ms = 10-20 MB/s
```

Negligible vs. PCIe 3.0 (8+ GB/s).

### 9.2 GPU Load

```
Compute utilization: ~20-40% (safe for desktop)
Memory utilization:  <1% (100 KB / 12 GB)
Thermal: Passive (no forced cooling needed)
```

### 9.3 Latency

```
Per-iteration:
  - GPU kernels:    1-2 ms
  - PCIe transfer:  <1 ms
  - Host control:   <1 ms
  - OS sleep:       1 ms
  Total:            2-5 ms (200-500 Hz)
```

---

## 10. Testing & Validation

### 10.1 Unit Tests (test.py)

1. **Engine init**: OpenCL context, kernel compilation
2. **Single step**: DFPM → FPIS → FDAΨ cycle
3. **Metrics**: Energy, variance, entropy calculation
4. **Checkpoint**: Save/load round-trip
5. **Stimulus**: Text injection verification
6. **JSON export**: Semantic snapshot generation
7. **Mini loop**: 100 steps of evolution

**Run**:
```bash
python test.py  # All 7 tests
```

### 10.2 Integration Tests

- **Network**: Send stimulus via TCP, verify injection
- **UI**: Slider updates, checkpoint trigger, export
- **Headless**: Run without UI, verify autonomous checkpointing
- **Multi-step**: 1000+ iterations, monitor stability

---

## 11. Extension Points

### 11.1 Audio Input

```python
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    amplitude = np.mean(np.abs(indata))
    pins += amplitude * 2.0  # Modulate frequency

sd.InputStream(callback=audio_callback).start()
```

### 11.2 Multi-GPU

```python
devices = cl.get_platforms()[0].get_devices()
# Replicate Ψ across devices
# Aggregate metrics
```

### 11.3 Formal Analysis

Export state trajectories for dynamical systems analysis:
- Lyapunov exponents
- Attractor basins
- Bifurcation diagrams

---

## 12. Limitations & Known Issues

1. **No quantum superposition** - Classical computation only.
2. **GPU-local only** - No multi-machine sync.
3. **No native visualization** - matplotlib optional (can lag).
4. **Windows OpenCL** - Vendor-specific quirks possible.
5. **No matrix acceleration** - Kernels are custom, not cuBLAS/rocBLAS.

---

## 13. Glossary

- **Ψ (psi)**: State field
- **DFPM**: Bulk Fractal Phase Modulation
- **FPIS**: Frequency Phase Identity Source
- **FDAΨ**: Feedback Differential Adaptation Psi
- **FHIDS**: Fractal Holographic Interference Distributed Storage
- **Pin**: Frequency identity unit (frequency "anchor")
- **Coupling**: Information flow between levels
- **Attractor**: Stable point in phase space
- **Autonomy**: Self-directed checkpointing

---

## 14. References

- OpenCL 2.x Specification (Khronos)
- AMD RDNA2 ISA Manual
- NumPy/PyOpenCL documentation
- Tkinter reference (Python stdlib)

---

**Document Status**: v3.0
**Date**: 2026-02-08
***Developed by: Vasile Lucian Borbeleac with the assistance of ChatGPT-5.2 and Claude Opus 4.6**
**Target Audience**: Developers, researchers, experimenters
