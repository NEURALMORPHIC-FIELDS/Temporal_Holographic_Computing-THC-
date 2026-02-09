![THC Logo](logo.png)

# Temporal Holographic Computation (THC)

**Developed by: Vasile Lucian Borbeleac with the assistance of ChatGPT-5.2 and Claude Opus 4.6**

**GPU-accelerated holographic dynamical system for reservoir computing**

THC is a complex-valued dynamical system where state vectors evolve as
psi in C^(P x D x Z), computed entirely on GPU via OpenCL targeting AMD
hardware (tested on RX 6700 XT). The engine drives 13 GPU kernels across
P=256 pins, D=16 dimensions, 3 hierarchical levels, and Z=64 holographic
planes. THC implements reservoir computing through Holographic Reduced
Representations (HRR) binding, providing proven nonlinear computation
capabilities with circular convolution in the frequency domain.

## Key Results

| Metric              | THC    | ESN Baseline |
|---------------------|--------|--------------|
| Memory Capacity     | 23.07  | 44.00        |
| XOR (matched tau)   | 100%   | 49%          |
| NARMA-10 R2         | 0.856  | 0.964        |
| NL-MC deg 2         | 0.98   | 0.00         |
| Pattern Recognition | 64.3%  | 81%          |

THC achieves perfect nonlinear separation on XOR tasks and strong
nonlinear memory capacity (NL-MC), demonstrating that holographic
dynamics encode information in qualitatively different ways than
conventional echo state networks.

## Features

- **Complex holographic dynamics** -- state evolution in C with leapfrog symplectic integration
- **HRR binding (Kernel 13)** -- circular convolution for compositional variable binding
- **Delay ring readout** -- temporal buffer enabling memory capacity measurement
- **Leapfrog integration** -- energy-preserving symplectic time stepping
- **Pin-field coupling** -- bidirectional interaction between pin frequencies and the bulk field
- **FHIDS memory** -- fractal holographic information density storage
- **Multi-delay binding** -- HRR binding across multiple time delays for temporal structure
- **Self-binding** -- state bound with its own delayed copy for nonlinear feature extraction

## Quick Start

### Prerequisites

- Python 3.10 or later
- AMD GPU with OpenCL 2.x runtime (AMD Adrenalin driver)
- PyOpenCL, NumPy

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# Launch with UI control panel
python main.py

# Headless mode
python main.py --headless
```

## Architecture

```
hfbp_gpu/
  engine.py          THCEngine - core GPU computation, state management
  kernels.cl         13 OpenCL kernels (DFPM, FPIS, FDA-psi, FHIDS,
                     coupling, HRR binding, delay ring, diagnostics)
  config.py          All system parameters (P, D, Z, dt, rates)
  main.py            Entry point and orchestration loop
  ui.py              Tkinter control panel for live parameter tuning
  visualizer.py      Real-time field and metric visualization
  network.py         TCP bridge for external stimulus injection
```

## Requirements

- **Python** 3.10+
- **PyOpenCL** with OpenCL 2.x backend
- **NumPy**
- **AMD GPU** with OpenCL runtime (tested: Radeon RX 6700 XT, 12 GB VRAM)

Optional: Matplotlib for extended visualization, SciPy for experiment analysis.

## Configuration

All parameters are defined in `config.py`. Key values:

| Parameter     | Default | Description                        |
|---------------|---------|------------------------------------|
| `P`           | 256     | Number of pins (frequency identities) |
| `D`           | 16      | Dimensions per pin                 |
| `LEVELS`      | 3       | Multi-scale hierarchy depth        |
| `Z`           | 64      | Holographic planes                 |
| `DT`          | 1.0     | Integration time step              |
| `TARGET_PIN`  | 80.0    | Attractor frequency                |
| `TUNE_RATE`   | 0.05    | Parameter update rate              |
| `ALPHA`       | 0.05    | Sub-level coupling strength        |

Modify `config.py` directly or override at runtime through the UI panel.

## Performance

- **GPU utilization**: 20--40% (desktop-safe continuous operation)
- **VRAM**: approximately 100 MB of 12 GB available
- **Kernel latency**: sub-millisecond per dispatch (AMD wavefront-optimized)

## License

Free for research, education, and non-commercial use. Commercial rights reserved.
See [LICENSE](../LICENSE) for details.

---

Tested on AMD Radeon RX 6700 XT, Windows 10/11, Python 3.10--3.12.
