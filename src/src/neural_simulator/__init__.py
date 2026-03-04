```python
"""
Neural Population Dynamics Simulator

A Monte Carlo simulation framework for stochastic firing patterns in neural populations.
"""

__version__ = "1.0.0"
__author__ = "Neural Dynamics Team"

from .models.neuron_models import PoissonNeuron, RefractoryNeuron
from .models.population_models import NeuralPopulation
from .simulation.gillespie import GillespieSimulator
from .data.allen_loader import AllenDataLoader
from .analysis.dynamics_analyzer import DynamicsAnalyzer

__all__ = [
    'PoissonNeuron',
    'RefractoryNeuron', 
    'NeuralPopulation',
    'GillespieSimulator',
    'AllenDataLoader',
    'DynamicsAnalyzer'
]
```
