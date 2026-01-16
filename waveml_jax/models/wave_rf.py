"""
WaveML-JAX: Wave RF - Receptive Field Model
============================================
Lightborne Intelligence

Wave-native convolutional model for signal processing.
Uses wave state representation with ERA throughout.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial
import flax.linen as nn

from ..core.representation import WaveState, encode_fft, decode_fft, total_energy
from ..core.invariants import InvariantBounds, DEFAULT_BOUNDS
from ..core.era_rectify import era_rectify


@jax.jit
def wave_conv(state: WaveState, 
              kernel_amp: jnp.ndarray,
              kernel_phase: jnp.ndarray) -> WaveState:
    """
    Wave convolution in frequency domain.
    
    Multiplication in frequency domain = convolution in time domain.
    """
    new_amp = state.amplitude * kernel_amp
    new_phase = state.phase + kernel_phase
    return WaveState(amplitude=new_amp, phase=new_phase)


@jax.jit
def wave_conv_with_era(state: WaveState,
                       kernel_amp: jnp.ndarray,
                       kernel_phase: jnp.ndarray,
                       bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """Wave convolution followed by ERA."""
    convolved = wave_conv(state, kernel_amp, kernel_phase)
    return era_rectify(convolved, bounds)


class WaveRFParams(NamedTuple):
    """Parameters for Wave RF layer."""
    kernel_amp: jnp.ndarray
    kernel_phase: jnp.ndarray
    bias_amp: jnp.ndarray


def init_waverf_params(key: jax.random.PRNGKey,
                       n_modes: int,
                       n_filters: int,
                       scale: float = 0.1) -> WaveRFParams:
    """Initialize Wave RF parameters."""
    k1, k2 = jax.random.split(key)
    
    return WaveRFParams(
        kernel_amp=jnp.ones((n_filters, n_modes)) + jax.random.normal(k1, (n_filters, n_modes)) * scale,
        kernel_phase=jax.random.normal(k2, (n_filters, n_modes)) * scale,
        bias_amp=jnp.zeros(n_filters)
    )


@jax.jit
def waverf_layer(state: WaveState,
                 params: WaveRFParams,
                 bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """
    Apply Wave RF layer.
    
    Each filter produces amplitude/phase modification, then aggregated.
    """
    n_filters = params.kernel_amp.shape[0]
    
    filtered_amps = state.amplitude[None, :] * params.kernel_amp
    filtered_phases = state.phase[None, :] + params.kernel_phase
    
    out_amp = jnp.mean(filtered_amps, axis=0) + jnp.mean(params.bias_amp)
    out_phase = jnp.mean(filtered_phases, axis=0)
    
    out_state = WaveState(
        amplitude=jnp.maximum(out_amp, 0.0),
        phase=out_phase
    )
    
    return era_rectify(out_state, bounds)


def init_waverf_stack(key: jax.random.PRNGKey,
                      n_modes: int,
                      n_filters: int,
                      n_layers: int) -> list:
    """Initialize stack of Wave RF layers."""
    params = []
    for i in range(n_layers):
        key, subkey = jax.random.split(key)
        params.append(init_waverf_params(subkey, n_modes, n_filters))
    return params


def waverf_forward(state: WaveState,
                   params_list: list,
                   bounds: InvariantBounds = DEFAULT_BOUNDS) -> WaveState:
    """Forward through Wave RF stack."""
    for params in params_list:
        state = waverf_layer(state, params, bounds)
    return state


class WaveRF(nn.Module):
    """
    Wave RF model for signal processing.
    
    Input signal -> FFT -> Wave layers + ERA -> IFFT -> Output signal
    """
    n_modes: int
    n_filters: int
    n_layers: int
    bounds: InvariantBounds = DEFAULT_BOUNDS
    
    @nn.compact
    def __call__(self, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            signal: Input signal, shape (length,)
        
        Returns:
            Processed signal, shape (length,)
        """
        length = signal.shape[-1]
        
        state = encode_fft(signal, self.n_modes)
        
        for i in range(self.n_layers):
            kernel_amp = self.param(
                f'kernel_amp_{i}',
                nn.initializers.ones,
                (self.n_filters, self.n_modes)
            )
            kernel_phase = self.param(
                f'kernel_phase_{i}',
                nn.initializers.normal(0.1),
                (self.n_filters, self.n_modes)
            )
            
            filtered_amp = state.amplitude[None, :] * kernel_amp
            filtered_phase = state.phase[None, :] + kernel_phase
            
            state = WaveState(
                amplitude=jnp.maximum(jnp.mean(filtered_amp, axis=0), 0.0),
                phase=jnp.mean(filtered_phase, axis=0)
            )
            state = era_rectify(state, self.bounds)
        
        return decode_fft(state, length)
