"""
Neural Population Dynamics Simulator

A Monte Carlo simulation framework for stochastic firing patterns in neural populations.
"""

__version__ = "1.0.0"
__author__ = "Kiran Shay"

from .models.neuron_models import PoissonNeuron, RefractoryNeuron
from .models.population_models import NeuralPopulation
from .simulation.gillespie import GillespieSimulator

__all__ = [
    'PoissonNeuron',
    'RefractoryNeuron',
    'NeuralPopulation',
    'GillespieSimulator',
]
