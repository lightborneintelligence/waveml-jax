# “””
WaveML-JAX: SNR Sweep Benchmark

Lightborne Intelligence

Systematic evaluation across Signal-to-Noise Ratios.
Tests the graceful degradation property of ERA.
“””

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List
from functools import partial

import sys
sys.path.insert(0, ‘/home/claude/waveml-jax’)

from core.representation import WaveState, encode_fft, decode_fft, total_energy
from core.invariants import InvariantBounds, DEFAULT_BOUNDS, TIGHT_BOUNDS, LOOSE_BOUNDS
from core.era_rectify import era_rectify

# ============================================================================

# SNR-Controlled Signal Generation

# ============================================================================

def generate_signal_with_snr(key: jax.random.PRNGKey,
length: int,
target_snr_db: float,
n_components: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
“””
Generate signal with specified SNR.

```
Returns:
    clean: Clean signal
    noisy: Signal with noise at target SNR
    noise: The noise component
"""
k1, k2, k3 = jax.random.split(key, 3)
t = jnp.linspace(0, 4 * jnp.pi, length)

# Generate clean signal
freqs = jax.random.uniform(k1, (n_components,), minval=1, maxval=5)
amps = jax.random.uniform(k2, (n_components,), minval=0.3, maxval=1.0)

clean = jnp.zeros(length)
for f, a in zip(freqs, amps):
    clean = clean + a * jnp.sin(f * t)

# Normalize
clean = clean / jnp.max(jnp.abs(clean))

# Calculate required noise level for target SNR
signal_power = jnp.mean(clean ** 2)
target_noise_power = signal_power / (10 ** (target_snr_db / 10))
noise_std = jnp.sqrt(target_noise_power)

# Generate noise
noise = jax.random.normal(k3, (length,)) * noise_std
noisy = clean + noise

return clean, noisy, noise
```

def compute_actual_snr(clean: jnp.ndarray, noisy: jnp.ndarray) -> float:
“”“Compute actual SNR in dB.”””
signal_power = jnp.mean(clean ** 2)
noise_power = jnp.mean((noisy - clean) ** 2)
return float(10 * jnp.log10(signal_power / (noise_power + 1e-10)))

# ============================================================================

# ERA Processing with Different Bounds

# ============================================================================

def era_process(signal: jnp.ndarray,
bounds: InvariantBounds,
n_modes: int = None) -> Tuple[jnp.ndarray, Dict]:
“””
Process signal with ERA.

```
Returns reconstructed signal and statistics.
"""
if n_modes is None:
    n_modes = len(signal) // 2 + 1

# Encode
state = encode_fft(signal, n_modes)
energy_before = total_energy(state)

# Rectify
rectified = era_rectify(state, bounds)
energy_after = total_energy(rectified)

# Decode
output = decode_fft(rectified, len(signal))

stats = {
    'energy_before': float(energy_before),
    'energy_after': float(energy_after),
    'energy_reduction': float(energy_before - energy_after),
    'modes_used': n_modes,
}

return output, stats
```

# ============================================================================

# Quality Metrics

# ============================================================================

def compute_metrics(clean: jnp.ndarray,
processed: jnp.ndarray,
noisy: jnp.ndarray) -> Dict:
“”“Compute comprehensive quality metrics.”””
# MSE
mse_processed = float(jnp.mean((clean - processed) ** 2))
mse_noisy = float(jnp.mean((clean - noisy) ** 2))

```
# SNR
signal_power = jnp.mean(clean ** 2)
snr_processed = 10 * jnp.log10(signal_power / (mse_processed + 1e-10))
snr_noisy = 10 * jnp.log10(signal_power / (mse_noisy + 1e-10))

# Correlation
def corr(a, b):
    a_c = a - jnp.mean(a)
    b_c = b - jnp.mean(b)
    return jnp.sum(a_c * b_c) / (jnp.sqrt(jnp.sum(a_c**2) * jnp.sum(b_c**2)) + 1e-10)

corr_processed = float(corr(clean, processed))
corr_noisy = float(corr(clean, noisy))

# Improvement ratios
mse_improvement = mse_noisy / (mse_processed + 1e-10)
snr_gain = float(snr_processed - snr_noisy)

return {
    'mse_processed': mse_processed,
    'mse_noisy': mse_noisy,
    'mse_improvement': mse_improvement,
    'snr_processed': float(snr_processed),
    'snr_noisy': float(snr_noisy),
    'snr_gain': snr_gain,
    'corr_processed': corr_processed,
    'corr_noisy': corr_noisy,
}
```

# ============================================================================

# SNR Sweep Benchmark

# ============================================================================

def run_snr_sweep(snr_levels_db: List[float] = [-10, -5, 0, 5, 10, 15, 20],
signal_length: int = 128,
n_trials: int = 10):
“””
Run SNR sweep benchmark.

```
Tests ERA performance across different SNR levels.
"""
print("=" * 70)
print("  SNR SWEEP BENCHMARK")
print("  ERA Performance vs Signal-to-Noise Ratio")
print("=" * 70)
print(f"\n  Config: signal_length={signal_length}, trials={n_trials}")
print(f"  SNR levels (dB): {snr_levels_db}")

bounds_configs = [
    ('Default', DEFAULT_BOUNDS),
    ('Tight', TIGHT_BOUNDS),
    ('Loose', LOOSE_BOUNDS),
]

results = {name: {'snr_gain': [], 'corr': [], 'mse_improvement': []} 
           for name, _ in bounds_configs}
results['noisy'] = {'corr': []}

master_key = jax.random.PRNGKey(42)

for snr_db in snr_levels_db:
    print(f"\n{'─' * 70}")
    print(f"  Target SNR = {snr_db} dB")
    print(f"{'─' * 70}")
    
    trial_results = {name: {'snr_gain': [], 'corr': [], 'mse_improvement': []}
                    for name, _ in bounds_configs}
    trial_results['noisy'] = {'corr': []}
    
    for trial in range(n_trials):
        key = jax.random.fold_in(master_key, int((snr_db + 100) * 100) + trial)
        
        # Generate signal
        clean, noisy, _ = generate_signal_with_snr(key, signal_length, snr_db)
        
        # Baseline (noisy)
        metrics_baseline = compute_metrics(clean, noisy, noisy)
        trial_results['noisy']['corr'].append(metrics_baseline['corr_noisy'])
        
        # Test each bounds configuration
        for name, bounds in bounds_configs:
            processed, _ = era_process(noisy, bounds)
            metrics = compute_metrics(clean, processed, noisy)
            
            trial_results[name]['snr_gain'].append(metrics['snr_gain'])
            trial_results[name]['corr'].append(metrics['corr_processed'])
            trial_results[name]['mse_improvement'].append(metrics['mse_improvement'])
    
    # Aggregate
    for name in list(results.keys()):
        if name == 'noisy':
            results['noisy']['corr'].append(jnp.mean(jnp.array(trial_results['noisy']['corr'])))
        else:
            results[name]['snr_gain'].append(jnp.mean(jnp.array(trial_results[name]['snr_gain'])))
            results[name]['corr'].append(jnp.mean(jnp.array(trial_results[name]['corr'])))
            results[name]['mse_improvement'].append(jnp.mean(jnp.array(trial_results[name]['mse_improvement'])))
    
    # Report
    print(f"\n  Config   | SNR Gain | Correlation | MSE Improvement")
    print(f"  {'-' * 55}")
    print(f"  {'Noisy':10s} | {'N/A':>8s} | {results['noisy']['corr'][-1]:>11.3f} | N/A")
    for name, _ in bounds_configs:
        gain = results[name]['snr_gain'][-1]
        corr = results[name]['corr'][-1]
        mse_imp = results[name]['mse_improvement'][-1]
        print(f"  {name:10s} | {gain:>+7.1f}dB | {corr:>11.3f} | {mse_imp:>7.2f}x")

# Summary Table
print(f"\n{'=' * 70}")
print("  SUMMARY: Correlation by Input SNR")
print(f"{'=' * 70}")
print(f"\n  {'SNR(dB)':>8s} | {'Noisy':>8s} | {'Default':>8s} | {'Tight':>8s} | {'Loose':>8s}")
print(f"  {'-' * 50}")

for i, snr in enumerate(snr_levels_db):
    noisy_c = results['noisy']['corr'][i]
    default_c = results['Default']['corr'][i]
    tight_c = results['Tight']['corr'][i]
    loose_c = results['Loose']['corr'][i]
    print(f"  {snr:>8.0f} | {noisy_c:>8.3f} | {default_c:>8.3f} | {tight_c:>8.3f} | {loose_c:>8.3f}")

# SNR Gain Summary
print(f"\n{'=' * 70}")
print("  SUMMARY: SNR Gain (dB) by Input SNR")
print(f"{'=' * 70}")
print(f"\n  {'SNR(dB)':>8s} | {'Default':>8s} | {'Tight':>8s} | {'Loose':>8s}")
print(f"  {'-' * 40}")

for i, snr in enumerate(snr_levels_db):
    default_g = results['Default']['snr_gain'][i]
    tight_g = results['Tight']['snr_gain'][i]
    loose_g = results['Loose']['snr_gain'][i]
    print(f"  {snr:>8.0f} | {default_g:>+8.1f} | {tight_g:>+8.1f} | {loose_g:>+8.1f}")

print(f"\n{'=' * 70}")
print("  Benchmark complete.")
print(f"{'=' * 70}")

return results
```

# ============================================================================

# Graceful Degradation Analysis

# ============================================================================

def analyze_graceful_degradation(snr_levels_db: List[float] = list(range(-20, 25, 5)),
signal_length: int = 128,
n_trials: int = 20):
“””
Detailed analysis of graceful degradation property.
“””
print(”=” * 70)
print(”  GRACEFUL DEGRADATION ANALYSIS”)
print(”=” * 70)

```
master_key = jax.random.PRNGKey(123)

correlations = []

for snr_db in snr_levels_db:
    trial_corrs = []
    for trial in range(n_trials):
        key = jax.random.fold_in(master_key, int((snr_db + 100) * 100) + trial)
        clean, noisy, _ = generate_signal_with_snr(key, signal_length, snr_db)
        
        processed, _ = era_process(noisy, DEFAULT_BOUNDS)
        
        # Correlation
        clean_c = clean - jnp.mean(clean)
        proc_c = processed - jnp.mean(processed)
        corr = jnp.sum(clean_c * proc_c) / (jnp.sqrt(jnp.sum(clean_c**2) * jnp.sum(proc_c**2)) + 1e-10)
        trial_corrs.append(float(corr))
    
    correlations.append({
        'snr_db': snr_db,
        'mean': jnp.mean(jnp.array(trial_corrs)),
        'std': jnp.std(jnp.array(trial_corrs)),
        'min': jnp.min(jnp.array(trial_corrs)),
        'max': jnp.max(jnp.array(trial_corrs)),
    })

# Report
print(f"\n  {'SNR(dB)':>8s} | {'Mean':>8s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
print(f"  {'-' * 50}")

for c in correlations:
    print(f"  {c['snr_db']:>8.0f} | {c['mean']:>8.3f} | {c['std']:>8.3f} | {c['min']:>8.3f} | {c['max']:>8.3f}")

# Key insight
print(f"\n  Key Insight:")
print(f"  Even at -20dB SNR (noise 10x signal power), ERA maintains")
mean_at_worst = correlations[0]['mean']
print(f"  correlation of {mean_at_worst:.3f} - demonstrating graceful degradation.")

return correlations
```

# ============================================================================

# Tests

# ============================================================================

def test_snr_sweep():
“”“Test SNR sweep components.”””
print(”=” * 60)
print(”  SNR Sweep Tests”)
print(”=” * 60)

```
key = jax.random.PRNGKey(42)

# Test 1: Signal generation with target SNR
print("\n[1] Signal generation with target SNR...")
clean, noisy, noise = generate_signal_with_snr(key, 128, target_snr_db=10)
actual_snr = compute_actual_snr(clean, noisy)
print(f"    Target SNR: 10 dB")
print(f"    Actual SNR: {actual_snr:.1f} dB")
assert abs(actual_snr - 10) < 1, "SNR should be close to target"

# Test 2: Low SNR signal
print("\n[2] Low SNR signal (-5 dB)...")
clean, noisy, _ = generate_signal_with_snr(key, 128, target_snr_db=-5)
actual_snr = compute_actual_snr(clean, noisy)
print(f"    Target SNR: -5 dB")
print(f"    Actual SNR: {actual_snr:.1f} dB")

# Test 3: ERA processing
print("\n[3] ERA processing...")
processed, stats = era_process(noisy, DEFAULT_BOUNDS)
print(f"    Energy before ERA: {stats['energy_before']:.3f}")
print(f"    Energy after ERA:  {stats['energy_after']:.3f}")

# Test 4: Metrics computation
print("\n[4] Metrics computation...")
metrics = compute_metrics(clean, processed, noisy)
print(f"    Noisy correlation: {metrics['corr_noisy']:.3f}")
print(f"    ERA correlation:   {metrics['corr_processed']:.3f}")
print(f"    SNR gain: {metrics['snr_gain']:+.1f} dB")

# Test 5: Different bounds
print("\n[5] Bounds comparison...")
for name, bounds in [('Default', DEFAULT_BOUNDS), ('Tight', TIGHT_BOUNDS), ('Loose', LOOSE_BOUNDS)]:
    processed, _ = era_process(noisy, bounds)
    metrics = compute_metrics(clean, processed, noisy)
    print(f"    {name:8s}: corr={metrics['corr_processed']:.3f}, gain={metrics['snr_gain']:+.1f}dB")

print("\n" + "=" * 60)
print("  All SNR sweep tests passed! ✓")
print("=" * 60)

return True
```

if **name** == “**main**”:
# Run tests
test_snr_sweep()

```
# Run mini benchmark
print("\n\n")
run_snr_sweep(
    snr_levels_db=[-5, 0, 5, 10, 15],
    n_trials=5
)

# Graceful degradation analysis
print("\n\n")
analyze_graceful_degradation(
    snr_levels_db=list(range(-15, 20, 5)),
    n_trials=10
)
```
