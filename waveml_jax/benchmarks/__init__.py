"""
WaveML Benchmarks
=================
Lightborne Intelligence

Benchmark tasks for evaluating ERA:
- delayed_copy: Long-horizon memory test
- noise_sweep: Robustness across noise levels
- snr_sweep: SNR-controlled evaluation
"""

from .delayed_copy import (
    generate_delayed_copy_task,
    generate_batch,
    accuracy_at_position,
    accuracy_after_delay,
    run_benchmark,
)

from .noise_sweep import (
    generate_clean_signal,
    add_noise,
    compute_mse,
    compute_snr,
    compute_correlation,
    process_with_era,
    run_noise_sweep,
)

from .snr_sweep import (
    generate_signal_with_snr,
    compute_actual_snr,
    era_process,
    compute_metrics,
    run_snr_sweep,
    analyze_graceful_degradation,
)

__all__ = [
    "generate_delayed_copy_task",
    "generate_batch",
    "accuracy_at_position",
    "accuracy_after_delay",
    "run_benchmark",
    "generate_clean_signal",
    "add_noise",
    "compute_mse",
    "compute_snr",
    "compute_correlation",
    "process_with_era",
    "run_noise_sweep",
    "generate_signal_with_snr",
    "compute_actual_snr",
    "era_process",
    "compute_metrics",
    "run_snr_sweep",
    "analyze_graceful_degradation",
]
