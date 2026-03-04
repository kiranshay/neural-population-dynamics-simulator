```python
"""
Gillespie algorithm implementation for exact stochastic simulation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from tqdm import tqdm

from ..models.population_models import NeuralPopulation

logger = logging.getLogger(__name__)


class GillespieSimulator:
    """
    Exact stochastic simulation using the Gillespie algorithm.
    
    The Gillespie algorithm provides exact simulation of stochastic processes
    by sampling from the correct probability distributions for event times
    and event types.
    
    Parameters:
    -----------
    population : NeuralPopulation
        The neural population to simulate
    dt_max : float
        Maximum time step for rate updates
    """
    
    def __init__(self, population: NeuralPopulation, dt_max: float = 0.001):
        self.population = population
        self.dt_max = dt_max
        
        # Simulation state
        self.current_time = 0.0
        self.event_count = 0
        
        # Data storage
        self.spike_times: Dict[int, List[float]] = {
            i: [] for i in range(len(population.neurons))
        }
        self.rate_history: List[Tuple[float, np.ndarray]] = []
        
        logger.info(f"Initialized Gillespie simulator with {len(population.neurons)} neurons")
    
    def simulate(self, total_time: float, save_rates: bool = True, 
                 rate_save_interval: float = 0.01) -> Dict[str, any]:
        """
        Run Gillespie simulation for specified duration.
        
        Parameters:
        -----------
        total_time : float
            Total simulation time (seconds)
        save_rates : bool
            Whether to save firing rates over time
        rate_save_interval : float
            Interval for saving population rates
            
        Returns:
        --------
        Dict[str, any]
            Simulation results containing spike times and statistics
        """
        logger.info(f"Starting Gillespie simulation for {total_time}s")
        
        self.current_time = 0.0
        self.event_count = 0
        next_rate_save = rate_save_interval
        
        # Progress bar
        pbar = tqdm(total=int(total_time * 1000), desc="Simulation", unit="ms")
        last_update_time = 0.0
        
        while self.current_time < total_time:
            # Get current population rates
            rates = self.population.get_population_rates(self.current_time)
            total_rate = np.sum(rates)
            
            # Save rates periodically
            if save_rates and self.current_time >= next_rate_save:
                self.rate_history.append((self.current_time, rates.copy()))
                next_rate_save += rate_save_interval
            
            # Check if any neurons can fire
            if total_rate <= 0:
                # No activity possible, advance time
                self.current_time = min(self.current_time + self.dt_max, total_time)
                continue
            
            # Sample time to next event
            tau = np.random.exponential(1.0 / total_rate)
            next_event_time = self.current_time + tau
            
            # Check if we exceed simulation time
            if next_event_time > total_time:
                self.current_time = total_time
                break
            
            # Select which neuron fires based on rates
            firing_probabilities = rates / total_rate
            firing_neuron = np.random.choice(
                len(self.population.neurons), 
                p=firing_probabilities
            )
            
            # Execute spike event
            self.current_time = next_event_time
            self._execute_spike(firing_neuron)
            self.event_count += 1
            
            # Update progress bar
            if self.current_time - last_update_time >= 0.001:  # Update every 1ms
                progress_ms = int(self.current_time * 1000)
                pbar.update(progress_ms - pbar.n)
                last_update_time = self.current_time
        
        pbar.close()
        
        # Compile results
        results = self._compile_results(total_time)
        logger.info(f"Simulation completed: {self.event_count} spikes in {total_time}s")
        
        return results
    
    def _execute_spike(self, neuron_id: int) -> None:
        """
        Execute a spike event.
        
        Parameters:
        -----------
        neuron_id : int
            ID of the firing neuron
        """
        # Record spike
        self.spike_times[neuron_id].append(self.current_time)
        self.population.record_spike(self.current_time, neuron_id)
        
        # Update neuron state
        self.population.neurons[neuron_id].update_state(self.current_time, 0.0)
    
    def _compile_results(self, total_time: float) -> Dict[str, any]:
        """Compile simulation results."""
        # Convert spike times to arrays
        spike_arrays = {nid: np.array(times) for nid, times in self.spike_times.items()}
        
        # Calculate basic statistics
        total_spikes = sum(len(times) for times in spike_arrays.values())
        mean_rate = total_spikes / (len(self.population.neurons) * total_time)
        
        # Per-neuron rates
        neuron_rates = np.array([
            len(spike_arrays[i]) / total_time 
            for i in range(len(self.population.neurons))
        ])
        
        # Population synchrony (coefficient of variation of ISIs)
        all_spike_times = []
        for times in spike_arrays.values():
            all_spike_times.extend(times)
        all_spike_times = np.sort(all_spike_times)
        
        if len(all_spike_times) > 1:
            isis = np.diff(all_spike_times)
            cv_isi = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
        else:
            cv_isi = 0
        
        # Rate history
        if self.rate_history:
            rate_times, rate_arrays = zip(*self.rate_history)
            rate_matrix = np.array(rate_arrays)
        else:
            rate_times = np.array([])
            rate_matrix = np.array([])
        
        return {
            'spike_times': spike_arrays,
            'total_time': total_time,
            'total_spikes': total_spikes,
            'mean_population_rate': mean_rate,
            'neuron_rates': neuron_rates,
            'cv_isi': cv_isi,
            'rate_times': np.array(rate_times),
            'rate_matrix': rate_matrix,
            'connectivity_stats': self.population.get_connectivity_stats(),
            'simulation_params': {
                'dt_max': self.dt_max,
                'event_count': self.event_count,
                'neuron_type': self.population.neuron_type,
                'population_params': self.population.params.__dict__
            }
        }
    
    def reset(self) -> None:
        """Reset simulator state."""
        self.current_time = 0.0
        self.event_count = 0
        self.spike_times = {i: [] for i in range(len(self.population.neurons))}
        self.rate_history = []
        self.population.reset()


class BatchSimulator:
    """
    Run multiple Gillespie simulations with different parameters.
    
    Useful for parameter sweeps and ensemble averaging.
    """
    
    def __init__(self, base_population_params: dict, base_sim_params: dict):
        self.base_population_params = base_population_params
        self.base_sim_params = base_sim_params
    
    def run_parameter_sweep(self, param_name: str, param_values: List[float],
                           n_trials: int = 5) -> Dict[str, List[Dict]]:
        """
        Run simulations across parameter values.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter to vary
        param_values : List[float]
            Values to test
        n_trials : int
            Number of trials per parameter value
            
        Returns:
        --------
        Dict[str, List[Dict]]
            Results organized by parameter value
        """
        results = {str(val): [] for val in param_values}
        
        for param_val in tqdm(param_values, desc="Parameter sweep"):
            for trial in range(n_trials):
                # Modify parameters
                pop_params = self.base_population_params.copy()
                if hasattr(pop_params, param_name):
                    setattr(pop_params, param_name, param_val)
                
                # Create population and simulator
                from ..models.population_models import PopulationParameters
                params = PopulationParameters(**pop_params)
                population = NeuralPopulation(params, seed=trial)
                simulator = GillespieSimulator(population)
                
                # Run simulation
                result = simulator.simulate(**self.base_sim_params)
                results[str(param_val)].append(result)
        
        return results
```
