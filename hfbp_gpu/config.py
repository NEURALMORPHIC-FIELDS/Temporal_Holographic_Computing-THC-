# ==================== THC CONFIG ====================
# Temporal Holographic Computation — AMD RX 6700 XT

# v3.0: COMPLEX PHASE HOLOGRAPHY
COMPLEX_MODE = True   # True = complex-valued state (float2/complex64), False = v2.x real mode

# PINI (Frequency Identity Units)
P = 256

# DIMENSIUNE PER PIN
D = 16

# NIVELURI MULTI-SCALE
LEVELS = 3

# HOLOGRAFIE (FHIDS planes) - FIX 3: was 8 (32:1 aliasing), now 64 (4:1)
Z = 64

# PARAMETRI EVOLUTIE
DT = 0.1             # Leapfrog CFL stability (was 0.5, caused blowup)
TARGET_PIN = 130.0   # Supercritical (k=1.3 > bifurcation at k=1.0)
TUNE_RATE = 0.002    # Very weak FPIS attractor (pins driven by field coupling, not target)
ALPHA = 0.05         # coupling between levels

# FIX 1: PIN-FIELD COUPLING
PIN_COUPLING_EPSILON = 0.5    # Strength of field -> pin feedback (must dominate FPIS)
PIN_COUPLING_RADIUS = 2       # Local ring coupling (2 neighbors each side = 5 pins, high locality)

# FIX 2: DELAY FEEDBACK
TAU = 16                      # Ring buffer slots (effective delay = 15 * DT = 1.5 time units)
DELAY_STRENGTH = 0.5          # Coefficient 'a' for delay term (sweep: chaos at 0.3-0.5)

# FIX 4: LEAPFROG INTEGRATION
VELOCITY_DAMPING = 0.005      # Near-conservative: allows sustained oscillations (sweep: best chaos at 0.0-0.005)

# STOCHASTIC DRIVING
NOISE_AMP = 0.01              # Per-step noise amplitude (symmetry breaking, sweep-validated)

# CONTROL ATRACTORI
TARGET_E = 0.002
STABLE_STEPS = 500

# SLEEP PENTRU UI (ms)
SLEEP_MS = 1

# PERSISTENTA
CHECKPOINT_DIR = "checkpoints"
AUTO_CHECKPOINT = True

# AUTONOMIE
AUTONOMY_ENABLED = True

# NETWORK
NETWORK_HOST = "127.0.0.1"
NETWORK_PORT = 7777

# LLM STIMULUS
LLM_GAIN = 0.005
LLM_ENABLED = True

# UI
UI_ENABLED = True
HEADLESS = False

# LOGGING
VERBOSE = True
