# Neural Population Dynamics Simulator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-red.svg)](https://neural-pop-simulator.streamlit.app)

> Monte Carlo simulation of stochastic neural population dynamics using the Gillespie algorithm

## Overview

This simulator models how populations of neurons generate coordinated spiking activity. It uses the **Gillespie algorithm** (exact stochastic simulation) to produce biologically plausible spike trains with:

- **Refractory periods** — neurons can't fire immediately after a spike (~2ms, matching cortical Na+ channel inactivation)
- **Correlated noise** — shared synaptic input creates pairwise correlations (via Cholesky decomposition)
- **Network coupling** — mean-field recurrent feedback drives emergent oscillations
- **External stimulation** — time-varying drive to probe network responses

## How It Works

### Gillespie Algorithm
Each neuron's instantaneous firing rate is:

```
λ_i(t) = λ₀ · S(t) · R(t - t_last) · exp(σ · η_i(t)) · (1 + α · A(t))
```

The algorithm samples exact spike times from exponential waiting time distributions — no discretization error.

### What You Can Explore
- **Asynchronous Irregular** state (ρ < 0.2, low connectivity) — the "awake cortex"
- **Synchronous Oscillatory** state (ρ > 0.5, high connectivity) — gamma rhythms
- **Pathological synchrony** (ρ > 0.8, high rate) — seizure-like dynamics

## Live Demo

**[Try it on Streamlit](https://neural-pop-simulator.streamlit.app)** — adjust parameters and watch population dynamics in real time.

## Running Locally

```bash
git clone https://github.com/kiranshay/neural-population-dynamics-simulator.git
cd neural-population-dynamics-simulator
pip install -r requirements.txt
streamlit run app.py
```

## Parameter Ranges

| Parameter | Range | Typical Cortex |
|-----------|-------|---------------|
| Base firing rate | 1-50 Hz | 1-20 Hz |
| Refractory period | 1-5 ms | 1-3 ms |
| Pairwise correlation | 0-1.0 | 0.1-0.3 |
| Connection probability | 0-0.5 | 0.1-0.2 |

## Limitations

- Network coupling uses a mean-field approximation (not individual synaptic connections)
- Population size limited to ~500 neurons in the Streamlit app for performance
- No GPU acceleration or tau-leaping for larger networks

## References

- Gillespie, D.T. (1977). Exact stochastic simulation of coupled chemical reactions. *J. Phys. Chem.*
- van Vreeswijk, C. & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitation and inhibition. *Science*
- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *J. Comput. Neurosci.*

## License

MIT License

---

Built by [Kiran Shay](https://kiranshay.github.io) · Johns Hopkins University
