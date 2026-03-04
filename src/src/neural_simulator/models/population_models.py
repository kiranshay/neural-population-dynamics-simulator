```python
"""
Neural population models for network-level dynamics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from .neuron_models import BaseNeuron, PoissonNeuron, RefractoryNeuron

logger = logging.getLogger(__name__)


@dataclass
class PopulationParameters:
    """Parameters for neural population."""
    n_neurons: int = 100
    connection_probability: float = 0.1
    weight_mean: float = 1.0
    weight_std: float = 0.2
    external_input_rate: float = 10.0
    noise_amplitude: float = 0.1


class NeuralPopulation:
    """
    A population of interconnected neurons.
    
    Parameters:
    -----------
    params : PopulationParameters
        Population configuration parameters
    neuron_type : str
        Type of neurons ('poisson', 'refractory', 'lif')
    seed : int
        Random seed for reproducibility
    """
    
    def __init__(self, params: PopulationParameters, 
                 neuron_type: str = 'poisson', seed: Optional[int] = None):
        self.params = params
        self.neuron_type = neuron_type
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize neurons
        self.neurons = self._create_neurons()
        
        # Initialize connectivity
        self.connectivity_matrix = self._create_connectivity_matrix()
        self.weight_matrix = self._create_weight_matrix()
        
        # State tracking
        self.spike_history: List[Tuple[float, int]] = []  # (time, neuron_id)
        self.current_time = 0.0
        
        logger.info(f"Created population with {len(self.neurons)} {neuron_type} neurons")
    
    def _create_neurons(self) -> List[BaseNeuron]:
        """Create list of neurons based on specified type."""
        neurons = []
        
        for i in range(self.params.n_neurons):
            if self.neuron_type == 'poisson':
                # Vary baseline rates
                base_rate = np.random.gamma(2.0, 2.5)  # Mean ~5 Hz
                neuron = PoissonNeuron(i, base_rate=base_rate)
            
            elif self.neuron_type == 'refractory':
                base_rate = np.random.gamma(2.0, 2.5)
                refractory = np.random.normal(0.002, 0.0005)  # ~2ms +/- 0.5ms
                refractory = max(0.001, refractory)  # Minimum 1ms
                neuron = RefractoryNeuron(i, base_rate=base_rate, 
                                        refractory_period=refractory)
            
            else:
                raise ValueError(f"Unknown neuron type: {self.neuron_type}")
            
            neurons.append(neuron)
        
        return neurons
    
    def _create_connectivity_matrix(self) -> np.ndarray:
        """Create binary connectivity matrix."""
        n = self.params.n_neurons
        connectivity = np.random.random((n, n)) < self.params.connection_probability
        
        # No self-connections
        np.fill_diagonal(connectivity, False)
        
        return connectivity
    
    def _create_weight_matrix(self) -> np.ndarray:
        """Create synaptic weight matrix."""
        n = self.params.n_neurons
        weights = np.random.normal(
            self.params.weight_mean, 
            self.params.weight_std, 
            (n, n)
        )
        
        # Apply connectivity mask
        weights = weights * self.connectivity_matrix.astype(float)
        
        # Ensure some inhibitory connections (20% of connections are inhibitory)
        inhibitory_mask = np.random.random((n, n)) < 0.2
        weights = np.where(inhibitory_mask & self.connectivity_matrix, 
                          -np.abs(weights), weights)
        
        return weights
    
    def compute_population_input(self, neuron_id: int, t: float) -> float:
        """
        Compute total input current to a neuron from network activity.
        
        Parameters:
        -----------
        neuron_id : int
            Target neuron ID
        t : float
            Current time
            
        Returns:
        --------
        float
            Total input current
        """
        # External Poisson input
        external_input = self.params.external_input_rate * np.random.exponential(
            1.0 / self.params.external_input_rate
        )
        
        # Synaptic input from network
        synaptic_input = 0.0
        tau_syn = 0.005  # 5ms synaptic time constant
        
        # Sum over recent spikes from connected neurons
        for spike_time, source_id in reversed(self.spike_history):
            if t - spike_time > 5 * tau_syn:  # Only consider recent spikes
                break
            
            if self.connectivity_matrix[source_id, neuron_id]:
                weight = self.weight_matrix[source_id, neuron_id]
                dt = t - spike_time
                synaptic_input += weight * np.exp(-dt / tau_syn)
        
        # Add correlated noise
        noise = np.random.normal(0, self.params.noise_amplitude)
        
        return external_input + synaptic_input + noise
    
    def get_population_rates(self, t: float) -> np.ndarray:
        """
        Get instantaneous firing rates for all neurons.
        
        Parameters:
        -----------
        t : float
            Current time
            
        Returns:
        --------
        np.ndarray
            Array of firing rates for each neuron
        """
        rates = np.zeros(len(self.neurons))
        
        for i, neuron in enumerate(self.neurons):
            input_current = self.compute_population_input(i, t)
            rates[i] = neuron.compute_rate(t, input_current)
        
        return rates
    
    def record_spike(self, t: float, neuron_id: int) -> None:
        """Record a spike event."""
        self.spike_history.append((t, neuron_id))
        self.neurons[neuron_id].last_spike_time = t
        
        # Keep only recent history (last 50ms)
        cutoff_time = t - 0.05
        self.spike_history = [(st, nid) for st, nid in self.spike_history 
                             if st > cutoff_time]
    
    def get_connectivity_stats(self) -> Dict[str, float]:
        """Get connectivity statistics."""
        return {
            'total_connections': int(self.connectivity_matrix.sum()),
            'connection_density': float(self.connectivity_matrix.mean()),
            'mean_in_degree': float(self.connectivity_matrix.sum(axis=0).mean()),
            'mean_out_degree': float(self.connectivity_matrix.sum(axis=1).mean()),
            'mean_weight': float(self.weight_matrix[self.connectivity_matrix].mean()),
            'weight_std': float(self.weight_matrix[self.connectivity_matrix].std()),
            'fraction_inhibitory': float((self.weight_matrix < 0).sum() / 
                                       self.connectivity_matrix.sum())
        }
    
    def reset(self) -> None:
        """Reset population state."""
        self.spike_history = []
        self.current_time = 0.0
        for neuron in self.neurons:
            neuron.last_spike_time = -np.inf
```
