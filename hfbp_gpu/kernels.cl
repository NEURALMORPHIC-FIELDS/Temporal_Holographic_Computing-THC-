// ==================== THC GPU KERNELS ====================
// Temporal Holographic Computation — AMD ROCm / OpenCL 2.x
// v3.0: Complex Phase Holographic Architecture
// Compile with -D COMPLEX_MODE for complex-valued state (float2)

// ============================================================
// COMPLEX HELPER FUNCTIONS (v3.0)
// ============================================================
#ifdef COMPLEX_MODE

// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
inline float2 c_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex magnitude: |a+bi| = sqrt(a^2 + b^2)
inline float c_abs(float2 z) {
    return sqrt(z.x * z.x + z.y * z.y);
}

// Phase-preserving tanh: tanh(|z|) * (z / |z|)
// Applies tanh to magnitude, preserves phase direction
// This is the core v3.0 nonlinearity: amplitude stabilizes, phase transports identity
inline float2 c_tanh_mag(float2 z) {
    float mag = c_abs(z);
    if (mag < 1e-8f) return (float2)(0.0f, 0.0f);
    float scale = tanh(mag) / mag;
    return (float2)(z.x * scale, z.y * scale);
}

// Magnitude clamp: if |z| > max_mag, scale down preserving phase
inline float2 c_clamp_mag(float2 z, float max_mag) {
    float mag = c_abs(z);
    if (mag > max_mag) {
        float scale = max_mag / mag;
        return (float2)(z.x * scale, z.y * scale);
    }
    return z;
}

#endif // COMPLEX_MODE


// ============================================================
// LEGACY KERNELS (kept for diagnostics / backward compat)
// These remain real-valued (float) regardless of COMPLEX_MODE
// ============================================================

// ---------- K1: DFPM (Bulk Fractal Phase Modulation) ----------
__kernel void dfpm(
    __global float* psi,
    __global float* pins,
    int D
) {
    int i = get_global_id(0);
    float inv_p = pins[i] / 100.0f;
    int base = i * D;
    for (int d = 0; d < D; d++) {
        float x = psi[base + d];
        psi[base + d] = tanh(x * inv_p);
    }
}

// ---------- K2: FPIS (Frequency Phase Identity Source) ----------
__kernel void fpis(
    __global float* pins,
    float target,
    float rate
) {
    int i = get_global_id(0);
    pins[i] += rate * (target - pins[i]);
}

// ---------- K3: FDA (Feedback Differential) ----------
__kernel void fda(
    __global float* psi,
    __global float* psi_prev,
    __global float* metric,
    float dt
) {
    int i = get_global_id(0);
    metric[i] = (psi[i] - psi_prev[i]) / dt;
}

// ---------- K7: STATE UPDATE (Euler) ----------
__kernel void update_state(
    __global float* psi,
    __global float* dpsi,
    float dt
) {
    int i = get_global_id(0);
    psi[i] += dpsi[i] * dt;
}

// ---------- K8: FUSED EVOLVE (legacy Euler) ----------
__kernel void evolve_level(
    __global float* psi,
    __global float* psi_prev,
    __global float* pins,
    __global float* dpsi,
    int D,
    float dt
) {
    int pid = get_global_id(0);
    int did = get_global_id(1);
    int idx = pid * D + did;

    float prev = psi[idx];
    psi_prev[idx] = prev;

    float p = pins[pid];
    float modulated = tanh(prev * (p / 100.0f));
    float deriv = (modulated - prev) / dt;
    dpsi[idx] = deriv;
    psi[idx] = modulated + deriv * dt;
}


// ============================================================
// CORE KERNELS v2.0/v3.0: Emergence-capable dynamics
// Dual-mode: float (real) or float2 (complex) via COMPLEX_MODE
// ============================================================

// ---------- K4: FHIDS STORE ----------
#ifdef COMPLEX_MODE
__kernel void fhids_store(
    __global float2* holo,
    __global float2* psi,
    int Z,
    int D
) {
    int i = get_global_id(0);
    holo[(i % Z) * D + (i % D)] = psi[i];
}
#else
__kernel void fhids_store(
    __global float* holo,
    __global float* psi,
    int Z,
    int D
) {
    int i = get_global_id(0);
    holo[(i % Z) * D + (i % D)] = psi[i];
}
#endif

// ---------- K5: FHIDS DIAG ----------
#ifdef COMPLEX_MODE
__kernel void fhids_diag(
    __global float2* holo,
    __global float2* out,
    int Z,
    int D
) {
    int i = get_global_id(0);
    out[i] = holo[(i % Z) * D + (i % D)];
}
#else
__kernel void fhids_diag(
    __global float* holo,
    __global float* out,
    int Z,
    int D
) {
    int i = get_global_id(0);
    out[i] = holo[(i % Z) * D + (i % D)];
}
#endif

// ---------- K6: MULTI-SCALE COUPLING ----------
#ifdef COMPLEX_MODE
__kernel void couple_levels(
    __global float2* psi_low,
    __global float2* psi_high,
    float alpha
) {
    int i = get_global_id(0);
    float2 lo = psi_low[i];
    float2 hi = psi_high[i];
    psi_low[i] = (float2)((1.0f - alpha) * lo.x + alpha * hi.x,
                           (1.0f - alpha) * lo.y + alpha * hi.y);
}
#else
__kernel void couple_levels(
    __global float* psi_low,
    __global float* psi_high,
    float alpha
) {
    int i = get_global_id(0);
    psi_low[i] = (1.0f - alpha) * psi_low[i] + alpha * psi_high[i];
}
#endif


// ---------- K9: LEAPFROG EVOLVE WITH DELAY ----------
// v3.0 complex: F = c_tanh_mag(k*psi) - psi - a*c_tanh_mag(psi_delay)
// Phase-preserving force: amplitude stabilizes, phase transports identity
// 2D dispatch: global_size = (P, D)
#ifdef COMPLEX_MODE
__kernel void evolve_leapfrog(
    __global float2* psi,          // complex state (read/write)
    __global float2* vel,          // complex velocity (read/write)
    __global float2* psi_delay,    // complex delayed state (read only)
    __global float* pins,          // pin frequencies - REAL (read only)
    __global float2* dpsi,         // complex state change for metrics
    int D,
    float dt,
    float delay_strength,
    float damping
) {
    int pid = get_global_id(0);
    int did = get_global_id(1);
    int idx = pid * D + did;

    float2 p = psi[idx];
    float2 v = vel[idx];
    float2 pd = psi_delay[idx];
    float k = pins[pid] / 100.0f;

    // Scale psi by real pin frequency: k * psi (real * complex)
    float2 kp = (float2)(k * p.x, k * p.y);

    // Phase-preserving force:
    //   tanh(|k*psi|) * (k*psi / |k*psi|)  -- fractal attractor
    //   - psi                                -- restoring force
    //   - a * tanh(|pd|) * (pd / |pd|)      -- bounded delay feedback
    float2 force = c_tanh_mag(kp) - p - delay_strength * c_tanh_mag(pd);

    // Leapfrog velocity update (symplectic-like)
    float2 v_new = (float2)((1.0f - damping) * v.x + dt * force.x,
                             (1.0f - damping) * v.y + dt * force.y);

    // Position update
    float2 p_new = (float2)(p.x + dt * v_new.x, p.y + dt * v_new.y);

    // Magnitude clamp: preserve phase, bound amplitude
    p_new = c_clamp_mag(p_new, 10.0f);
    v_new = c_clamp_mag(v_new, 10.0f);

    // Write back
    vel[idx] = v_new;
    psi[idx] = p_new;
    dpsi[idx] = (float2)(p_new.x - p.x, p_new.y - p.y);
}
#else
__kernel void evolve_leapfrog(
    __global float* psi,
    __global float* vel,
    __global float* psi_delay,
    __global float* pins,
    __global float* dpsi,
    int D,
    float dt,
    float delay_strength,
    float damping
) {
    int pid = get_global_id(0);
    int did = get_global_id(1);
    int idx = pid * D + did;

    float p = psi[idx];
    float v = vel[idx];
    float pd = psi_delay[idx];
    float k = pins[pid] / 100.0f;

    float force = tanh(k * p) - p - delay_strength * tanh(pd);
    float v_new = (1.0f - damping) * v + dt * force;
    float p_new = p + dt * v_new;

    p_new = clamp(p_new, -10.0f, 10.0f);
    v_new = clamp(v_new, -10.0f, 10.0f);

    vel[idx] = v_new;
    psi[idx] = p_new;
    dpsi[idx] = p_new - p;
}
#endif


// ---------- K10: PIN-FIELD COUPLING ----------
// Pins remain real. In complex mode, uses magnitude of field for coupling.
// 1D dispatch: global_size = (P,)
#ifdef COMPLEX_MODE
__kernel void pin_field_coupling(
    __global float* pins,          // pin frequencies - REAL (read/write)
    __global float2* psi,          // complex field state (read only)
    int D,
    int P_size,
    int radius,
    float epsilon,
    float target,
    float fpis_rate
) {
    int i = get_global_id(0);

    float field_sum = 0.0f;
    int count = 0;

    for (int j = -radius; j <= radius; j++) {
        int jj = ((i + j) % P_size + P_size) % P_size;

        float pin_field = 0.0f;
        for (int d = 0; d < D; d++) {
            // Use magnitude as field strength (phase-invariant coupling)
            pin_field += c_abs(psi[jj * D + d]);
        }
        field_sum += pin_field / (float)D;
        count++;
    }

    float mean_field = field_sum / (float)count;
    pins[i] += fpis_rate * (target - pins[i]) + epsilon * tanh(mean_field);
}
#else
__kernel void pin_field_coupling(
    __global float* pins,
    __global float* psi,
    int D,
    int P_size,
    int radius,
    float epsilon,
    float target,
    float fpis_rate
) {
    int i = get_global_id(0);

    float field_sum = 0.0f;
    int count = 0;

    for (int j = -radius; j <= radius; j++) {
        int jj = ((i + j) % P_size + P_size) % P_size;

        float pin_field = 0.0f;
        for (int d = 0; d < D; d++) {
            pin_field += psi[jj * D + d];
        }
        field_sum += pin_field / (float)D;
        count++;
    }

    float mean_field = field_sum / (float)count;
    pins[i] += fpis_rate * (target - pins[i]) + epsilon * tanh(mean_field);
}
#endif


// ---------- K11: COPY BUFFER ----------
// GPU-side buffer copy (delay ring rotation)
#ifdef COMPLEX_MODE
__kernel void copy_buffer(
    __global float2* dst,
    __global float2* src
) {
    int i = get_global_id(0);
    dst[i] = src[i];
}
#else
__kernel void copy_buffer(
    __global float* dst,
    __global float* src
) {
    int i = get_global_id(0);
    dst[i] = src[i];
}
#endif


// ---------- K12: ADD NOISE / INPUT INJECTION ----------
// In complex mode: noise is real-valued, injected into Re(psi) only
// Phase evolves purely from dynamics (not driven by noise)
#ifdef COMPLEX_MODE
__kernel void add_noise(
    __global float2* psi,
    __global float* noise     // REAL noise buffer (not complex)
) {
    int i = get_global_id(0);
    psi[i].x += noise[i];    // inject into real part only
}
#else
__kernel void add_noise(
    __global float* psi,
    __global float* noise
) {
    int i = get_global_id(0);
    psi[i] += noise[i];
}
#endif


// ============================================================
// HOLOGRAPHIC BINDING v3.0 (Complex Circular Convolution)
// ============================================================

// ---------- K13: HOLOGRAPHIC BIND ----------
// v3.0: Complex circular convolution using c_mul
// bind(A, B)[d] = (1/sqrt(D)) * sum_k c_mul(A[k], B[(d-k)%D])
// 2D dispatch: global_size = (P, D)
#ifdef COMPLEX_MODE
__kernel void holographic_bind(
    __global float2* A,
    __global float2* B,
    __global float2* C,
    int D
) {
    int pid = get_global_id(0);
    int did = get_global_id(1);
    int base = pid * D;

    float2 sum = (float2)(0.0f, 0.0f);
    for (int k = 0; k < D; k++) {
        sum += c_mul(A[base + k], B[base + ((did - k + D) % D)]);
    }
    float norm = sqrt((float)D);
    C[base + did] = (float2)(sum.x / norm, sum.y / norm);
}
#else
__kernel void holographic_bind(
    __global float* A,
    __global float* B,
    __global float* C,
    int D
) {
    int pid = get_global_id(0);
    int did = get_global_id(1);
    int base = pid * D;

    float sum = 0.0f;
    for (int k = 0; k < D; k++) {
        sum += A[base + k] * B[base + ((did - k + D) % D)];
    }
    C[base + did] = sum / sqrt((float)D);
}
#endif
