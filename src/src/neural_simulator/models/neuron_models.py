```python
"""
Individual neuron models for stochastic simulation.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from numba import jit


class BaseNeuron(ABC):
    """Abstract base class for neuron models."""
    
    def __init__(self, neuron_id: int):
        self.neuron_id = neuron_id
        self.last_spike_time: float = -np.inf
        self.parameters: Dict[str, float] = {}
    
    @abstractmethod
    def compute_rate(self, t: float, input_current: float) -> float:
        """Compute instantaneous firing rate."""
        pass
    
    @abstractmethod
    def update_state(self, t: float, dt: float) -> None:
        """Update neuron state variables."""
        pass


class PoissonNeuron(BaseNeuron):
    """
    Simple Poisson neuron with exponential ISI distribution.
    
    Parameters:
    -----------
    neuron_id : int
        Unique identifier for the neuron
    base_rate : float
        Baseline firing rate in Hz
    gain : float
        Input gain parameter
    """
    
    def __init__(self, neuron_id: int, base_rate: float = 5.0, gain: float = 1.0):
        super().__init__(neuron_id)
        self.base_rate = base_rate
        self.gain = gain
        self.parameters = {
            'base_rate': base_rate,
            'gain': gain
        }
    
    def compute_rate(self, t: float, input_current: float) -> float:
        """
        Compute instantaneous firing rate.
        
        Parameters:
        -----------
        t : float
            Current time
        input_current : float
            Total input current
            
        Returns:
        --------
        float
            Instantaneous firing rate in Hz
        """
        return max(0.0, self.base_rate + self.gain * input_current)
    
    def update_state(self, t: float, dt: float) -> None:
        """Update neuron state (no additional state for Poisson)."""
        pass


class RefractoryNeuron(BaseNeuron):
    """
    Neuron with absolute refractory period.
    
    Parameters:
    -----------
    neuron_id : int
        Unique identifier for the neuron
    base_rate : float
        Baseline firing rate in Hz
    gain : float
        Input gain parameter
    refractory_period : float
        Absolute refractory period in seconds
    """
    
    def __init__(self, neuron_id: int, base_rate: float = 5.0, 
                 gain: float = 1.0, refractory_period: float = 0.002):
        super().__init__(neuron_id)
        self.base_rate = base_rate
        self.gain = gain
        self.refractory_period = refractory_period
        self.parameters = {
            'base_rate': base_rate,
            'gain': gain,
            'refractory_period': refractory_period
        }
    
    def compute_rate(self, t: float, input_current: float) -> float:
        """
        Compute instantaneous firing rate with refractory period.
        
        Parameters:
        -----------
        t : float
            Current time
        input_current : float
            Total input current
            
        Returns:
        --------
        float
            Instantaneous firing rate in Hz (0 if in refractory period)
        """
        if t - self.last_spike_time < self.refractory_period:
            return 0.0
        return max(0.0, self.base_rate + self.gain * input_current)
    
    def update_state(self, t: float, dt: float) -> None:
        """Update neuron state."""
        pass


@jit(nopython=True)
def _compute_synaptic_input(weights: np.ndarray, spike_times: np.ndarray, 
                           t: float, tau_syn: float) -> float:
    """
    Compute synaptic input using exponential decay.
    
    Parameters:
    -----------
    weights : np.ndarray
        Synaptic weights
    spike_times : np.ndarray
        Times of recent spikes
    t : float
        Current time
    tau_syn : float
        Synaptic time constant
        
    Returns:
    --------
    float
        Total synaptic input
    """
    total_input = 0.0
    for i in range(len(spike_times)):
        if spike_times[i] > 0 and t > spike_times[i]:
            dt = t - spike_times[i]
            total_input += weights[i] * np.exp(-dt / tau_syn)
    return total_input


class LeakyIntegrateFireNeuron(BaseNeuron):
    """
    Leaky integrate-and-fire neuron with exponential synaptic currents.
    
    Parameters:
    -----------
    neuron_id : int
        Unique identifier
    tau_m : float
        Membrane time constant (s)
    v_rest : float
        Resting potential (mV)
    v_thresh : float
        Spike threshold (mV)
    v_reset : float
        Reset potential (mV)
    tau_syn : float
        Synaptic time constant (s)
    """
    
    def __init__(self, neuron_id: int, tau_m: float = 0.02, 
                 v_rest: float = -65.0, v_thresh: float = -50.0,
                 v_reset: float = -65.0, tau_syn: float = 0.005):
        super().__init__(neuron_id)
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_syn = tau_syn
        
        # State variables
        self.v_membrane = v_rest
        self.synaptic_weights = np.array([])
        self.recent_spike_times = np.array([])
        
        self.parameters = {
            'tau_m': tau_m,
            'v_rest': v_rest,
            'v_thresh': v_thresh,
            'v_reset': v_reset,
            'tau_syn': tau_syn
        }
    
    def compute_rate(self, t: float, input_current: float) -> float:
        """
        Compute instantaneous firing rate based on distance to threshold.
        
        Uses exponential approximation for crossing probability.
        """
        if len(self.synaptic_weights) > 0:
            synaptic_input = _compute_synaptic_input(
                self.synaptic_weights, self.recent_spike_times, t, self.tau_syn
            )
        else:
            synaptic_input = 0.0
        
        # Total input
        total_input = input_current + synaptic_input
        
        # Steady-state voltage
        v_ss = self.v_rest + total_input * self.tau_m
        
        if v_ss <= self.v_thresh:
            return 0.0
        
        # Approximate firing rate based on voltage dynamics
        rate = (v_ss - self.v_thresh) / (self.tau_m * (self.v_thresh - self.v_rest))
        return max(0.0, rate * 1000.0)  # Convert to Hz
    
    def update_state(self, t: float, dt: float) -> None:
        """Update membrane voltage."""
        # Simple Euler integration would go here
        # For simplicity, we'll use the rate-based approximation
        pass
    
    def set_synaptic_connections(self, weights: np.ndarray):
        """Set synaptic connection weights."""
        self.synaptic_weights = weights.copy()
        self.recent_spike_times = np.zeros(len(weights))
    
    def receive_spike(self, source_id: int, spike_time: float):
        """Record incoming spike for synaptic integration."""
        if source_id < len(self.recent_spike_times):
            self.recent_spike_times[source_id] = spike_time
```
