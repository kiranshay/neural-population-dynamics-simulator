import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import odeint
import time

# Page configuration
st.set_page_config(
    page_title="Neural Population Dynamics Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.03);
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px;
        background: transparent;
        border-radius: 12px;
        padding: 0 24px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.08);
        color: #667eea;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .metric-container h3 {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }

    .metric-container p {
        font-size: 0.85rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 500;
    }

    /* Theory Box */
    .theory-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        margin: 1.25rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .theory-box h4 {
        font-family: 'Inter', sans-serif;
        color: #334155;
        margin-top: 0;
        font-weight: 600;
    }

    .theory-box h5 {
        font-family: 'Inter', sans-serif;
        color: #475569;
        margin-top: 1rem;
        font-weight: 600;
    }

    /* Section Headers - Enhanced */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        display: block;
    }

    /* Subsection Headers */
    .subsection-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #334155;
        margin: 2rem 0 1rem 0;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
    }

    /* Ensure all text inside expanders is readable */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li,
    [data-testid="stExpander"] span {
        color: #1e293b !important;
    }

    [data-testid="stExpander"] strong {
        color: #0f172a !important;
    }

    [data-testid="stExpander"] em {
        color: #334155 !important;
    }

    /* Content cards inside expanders */
    .concept-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .concept-card h5 {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #334155 !important;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .concept-card p, .concept-card li {
        color: #1e293b !important;
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }

    /* Highlight box */
    .highlight-box {
        background: #f0f0ff;
        border: 2px solid #818cf8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .highlight-box p {
        color: #1e1b4b !important;
        font-weight: 500;
        margin: 0;
        line-height: 1.6;
    }

    .highlight-box strong {
        color: #1e1b4b !important;
    }

    /* Key point callout */
    .key-point {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .key-point-icon {
        font-size: 1.25rem;
        flex-shrink: 0;
    }

    .key-point p {
        color: #065f46 !important;
        font-weight: 500;
        margin: 0;
        line-height: 1.6;
    }

    .key-point strong {
        color: #065f46 !important;
    }

    /* Definition list styling */
    .def-item {
        display: flex;
        margin-bottom: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f1f5f9;
    }

    .def-term {
        font-weight: 600;
        color: #667eea;
        min-width: 140px;
        flex-shrink: 0;
    }

    .def-desc {
        color: #1e293b !important;
        line-height: 1.5;
    }

    /* Algorithm steps */
    .algo-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px dashed #e2e8f0;
    }

    .algo-step:last-child {
        border-bottom: none;
    }

    .step-num {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .step-content {
        color: #1e293b !important;
        line-height: 1.6;
    }

    .step-content strong {
        color: #0f172a !important;
    }

    /* Parameter cards */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .param-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
    }

    .param-card h6 {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: #667eea;
        margin: 0 0 0.5rem 0;
    }

    .param-card p {
        color: #334155 !important;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    /* State cards for network regimes */
    .state-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: all 0.2s ease;
    }

    .state-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .state-card h4 {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0 0 0.25rem 0;
    }

    .state-card .state-subtitle {
        font-style: italic;
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .state-card .params {
        background: #e0e7ff;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.75rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #3730a3;
    }

    .state-card .props {
        color: #1e293b !important;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Table styling enhancement */
    .stMarkdown table {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .stMarkdown thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        text-align: left;
    }

    .stMarkdown tbody td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b;
    }

    .stMarkdown tbody tr:last-child td {
        border-bottom: none;
    }

    .stMarkdown tbody tr:nth-child(even) {
        background: #f8fafc;
    }

    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
    }

    /* Info Box */
    .stAlert {
        border-radius: 12px;
        border: none;
    }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .footer p {
        color: #64748b;
        font-family: 'Inter', sans-serif;
        margin: 0.5rem 0;
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }

    .footer a:hover {
        color: #764ba2;
    }

    /* Make plots responsive */
    .stPlotlyChart, [data-testid="stImage"], .stPyplot > div {
        max-width: 100%;
        overflow-x: auto;
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-header {
            font-size: 1.5rem;
            line-height: 1.3;
        }

        .subtitle {
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .metric-container {
            padding: 0.75rem;
            border-radius: 12px;
            margin-bottom: 0.5rem;
        }

        .metric-container h3 {
            font-size: 1.1rem;
        }

        .metric-container p {
            font-size: 0.75rem;
        }

        .section-header {
            font-size: 1.2rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
        }

        .highlight-box {
            padding: 0.75rem;
            font-size: 0.85rem;
        }

        .highlight-box p {
            font-size: 0.85rem;
        }

        .param-grid {
            grid-template-columns: 1fr;
        }

        .def-item {
            flex-direction: column;
            gap: 0.25rem;
        }

        .def-term {
            min-width: auto;
        }

        .algo-step {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            padding: 4px;
            gap: 4px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 36px;
            padding: 0 10px;
            font-size: 0.75rem;
        }

        .concept-card {
            padding: 0.75rem;
        }

        .state-card {
            padding: 0.75rem;
        }

        .subsection-header {
            font-size: 1.1rem;
            padding: 0.5rem 0.75rem;
        }

        .footer {
            padding: 1rem;
        }

        .footer p {
            font-size: 0.8rem;
        }
    }

    /* Button Styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Neural Population Dynamics Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Monte Carlo Simulation of Stochastic Neural Networks</div>', unsafe_allow_html=True)

# Mobile hint — sidebar is collapsed by default
st.markdown("""
<div class="highlight-box" style="margin-bottom: 1rem;">
<p>👈 <strong>Getting started:</strong> Open the sidebar (arrow at top-left)
to configure simulation parameters, then click <strong>Run Simulation</strong>.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## 🎛️ Simulation Parameters")

# Population parameters
st.sidebar.markdown("### Population Settings")
N_neurons = st.sidebar.slider("Number of Neurons", 50, 500, 200, 50)
simulation_time = st.sidebar.slider("Simulation Time (ms)", 100, 2000, 1000, 100)
with st.sidebar.expander("ℹ️ What are these?"):
    st.markdown("""
    **Number of Neurons** — How many neurons are in the simulated population.
    Real cortical columns contain ~10,000 neurons; we simulate a subset for speed.

    **Simulation Time** — How long to record activity (in milliseconds).
    1000ms = 1 second of neural activity. Typical experiments record 1-10 seconds.
    """)

# Neural parameters
st.sidebar.markdown("### Neural Properties")
base_firing_rate = st.sidebar.slider("Base Firing Rate (Hz)", 1.0, 50.0, 10.0, 1.0)
refractory_period = st.sidebar.slider("Refractory Period (ms)", 1.0, 5.0, 2.0, 0.5)
noise_strength = st.sidebar.slider("Synaptic Noise", 0.0, 2.0, 0.5, 0.1)
with st.sidebar.expander("ℹ️ What are these?"):
    st.markdown("""
    **Base Firing Rate** — How often each neuron spikes *on average* without any stimulus.
    Cortical neurons typically fire at 1-20 Hz. Higher = more active baseline.

    **Refractory Period** — After firing, a neuron *cannot* fire again for this duration
    (Na⁺ channels need to recover). Cortical neurons: ~1-3 ms.

    **Synaptic Noise** — Random fluctuations in firing rate from background synaptic input.
    At 0 = perfectly regular firing. At 1-2 = highly variable, more biologically realistic.
    """)

# Network parameters
st.sidebar.markdown("### Network Dynamics")
connectivity = st.sidebar.slider("Network Connectivity", 0.0, 0.5, 0.1, 0.05)
correlation_strength = st.sidebar.slider("Population Correlation", 0.0, 1.0, 0.3, 0.1)
with st.sidebar.expander("ℹ️ What are these?"):
    st.markdown("""
    **Network Connectivity** — Strength of recurrent feedback. When neurons fire, they
    influence other neurons' firing rates. Higher values → oscillations and synchrony.
    Real cortex: ~0.1-0.2 connection probability.

    **Population Correlation** — How much neurons share common input fluctuations.
    At 0 = independent. At 0.3 = typical cortical correlation (shared synaptic input).
    At 0.8+ = highly synchronized (pathological, seizure-like).
    """)

# Stimulus parameters
st.sidebar.markdown("### External Stimulus")
stimulus_start = st.sidebar.slider("Stimulus Start (ms)", 100, 800, 300, 50)
stimulus_duration = st.sidebar.slider("Stimulus Duration (ms)", 50, 500, 200, 50)
stimulus_strength = st.sidebar.slider("Stimulus Strength", 0.0, 3.0, 1.0, 0.1)
with st.sidebar.expander("ℹ️ What are these?"):
    st.markdown("""
    **Stimulus Start/Duration** — When and how long the external drive is applied.
    Think of this as a sensory stimulus (like a flash of light) that excites the population.

    **Stimulus Strength** — How much the stimulus boosts firing rates.
    At 1.0 = doubles the rate during stimulus. At 0 = no stimulus.
    This mimics how sensory input increases neural excitability.
    """)

class NeuralPopulationSimulator:
    def __init__(self, N, T, dt=0.1):
        self.N = N  # Number of neurons
        self.T = T  # Total time (ms)
        self.dt = dt  # Time step (ms)
        self.t = np.arange(0, T, dt)
        self.spike_times = [[] for _ in range(N)]
        self.spike_trains = np.zeros((N, len(self.t)))
        
    def gillespie_step(self, rates, t_current):
        """Single step of Gillespie algorithm for exact stochastic simulation"""
        total_rate = np.sum(rates)
        if total_rate == 0:
            return np.inf, -1
        
        # Time to next event
        dt_event = np.random.exponential(1.0 / total_rate)
        
        # Which neuron fires
        neuron_idx = np.random.choice(self.N, p=rates / total_rate)
        
        return dt_event, neuron_idx
    
    def simulate(self, base_rate, refractory, noise, connectivity, 
                correlation, stim_start, stim_dur, stim_strength):
        """Run the full population simulation"""
        
        # Initialize
        last_spike_time = np.full(self.N, -np.inf)
        population_rate = np.zeros(len(self.t))
        
        # Generate correlated noise
        if correlation > 0:
            # Clamp correlation slightly below 1 to ensure positive-definite matrix
            rho = min(correlation, 0.999)
            correlation_matrix = rho * np.ones((self.N, self.N)) + (1-rho) * np.eye(self.N)
            L = np.linalg.cholesky(correlation_matrix)
        
        t_idx = 0
        t_current = 0
        
        with st.spinner("Running Gillespie simulation..."):
            progress_bar = st.progress(0)
            
            while t_current < self.T and t_idx < len(self.t):
                # Update progress
                progress_bar.progress(min(t_current / self.T, 1.0))
                
                # Calculate instantaneous firing rates
                rates = np.full(self.N, base_rate / 1000.0)  # Convert Hz to per ms
                
                # Apply refractory period
                refractory_mask = (t_current - last_spike_time) < refractory
                rates[refractory_mask] = 0
                
                # Add stimulus
                if stim_start <= t_current <= stim_start + stim_dur:
                    rates *= (1 + stim_strength)
                
                # Add noise and network effects
                if correlation > 0:
                    noise_vec = np.random.normal(0, noise, self.N)
                    correlated_noise = L @ noise_vec
                    rates *= np.exp(correlated_noise)
                else:
                    rates *= np.exp(np.random.normal(0, noise, self.N))
                
                # Network coupling (mean-field approximation)
                # Count spikes in last 10ms window across all neurons
                recent_activity = sum(
                    1 for spikes in self.spike_times
                    for s in reversed(spikes)
                    if t_current - 10 < s <= t_current
                ) if connectivity > 0 else 0
                network_effect = connectivity * recent_activity / self.N
                rates *= (1 + network_effect)
                
                rates = np.maximum(rates, 0)  # Ensure non-negative
                
                # Gillespie step
                dt_event, neuron_idx = self.gillespie_step(rates, t_current)
                
                if dt_event == np.inf or neuron_idx == -1:
                    break
                
                # Advance time
                t_current += dt_event
                
                # Record spike
                if neuron_idx >= 0 and t_current < self.T:
                    self.spike_times[neuron_idx].append(t_current)
                    last_spike_time[neuron_idx] = t_current
                    
                    # Update spike trains
                    spike_idx = int(t_current / self.dt)
                    if spike_idx < len(self.t):
                        self.spike_trains[neuron_idx, spike_idx] += 1
                
                # Update time index
                while t_idx < len(self.t) and self.t[t_idx] <= t_current:
                    t_idx += 1
            
            progress_bar.empty()
        
        # Calculate population firing rate
        window_size = int(50 / self.dt)  # 50ms window
        for i in range(len(self.t)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(self.t), i + window_size//2)
            population_rate[i] = np.sum(self.spike_trains[:, start_idx:end_idx]) / (window_size * self.dt / 1000.0) / self.N
        
        return population_rate

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    # Create simulator
    simulator = NeuralPopulationSimulator(N_neurons, simulation_time)
    
    # Run simulation
    pop_rate = simulator.simulate(
        base_firing_rate, refractory_period, noise_strength,
        connectivity, correlation_strength, stimulus_start, 
        stimulus_duration, stimulus_strength
    )
    
    # Store results in session state
    st.session_state.simulator = simulator
    st.session_state.pop_rate = pop_rate
    st.session_state.simulation_done = True

# Main content tabs
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False

tab1, tab2, tab3, tab4 = st.tabs(["📊 Population Dynamics", "🎯 Raster Plot", "📈 Statistics", "🧮 Theory"])

with tab1:
    st.markdown('<p class="section-header">📊 Population Firing Rate Dynamics</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>📊 <strong>What this shows:</strong> The average firing rate across all neurons over time.
    The red shaded region marks when the stimulus is active — you should see the rate increase there.
    The bottom plot compares the stochastic simulation against a deterministic mean-field prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.simulation_done:
        simulator = st.session_state.simulator
        pop_rate = st.session_state.pop_rate
        
        # Create main plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Population rate
        ax1.plot(simulator.t, pop_rate, 'b-', linewidth=2, alpha=0.8)
        ax1.axvspan(stimulus_start, stimulus_start + stimulus_duration, 
                   alpha=0.3, color='red', label='Stimulus')
        ax1.set_ylabel('Population Rate (Hz)', fontsize=12)
        ax1.set_title('Population-Level Dynamics', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mean field approximation
        mean_field_rate = base_firing_rate * np.ones_like(simulator.t)
        stim_mask = (simulator.t >= stimulus_start) & (simulator.t <= stimulus_start + stimulus_duration)
        mean_field_rate[stim_mask] *= (1 + stimulus_strength)
        
        ax2.plot(simulator.t, pop_rate, 'b-', linewidth=2, label='Simulated', alpha=0.8)
        ax2.plot(simulator.t, mean_field_rate, 'r--', linewidth=2, label='Mean Field')
        ax2.axvspan(stimulus_start, stimulus_start + stimulus_duration, 
                   alpha=0.3, color='red')
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Rate (Hz)', fontsize=12)
        ax2.set_title('Comparison with Mean Field Theory', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display key metrics
        total_spikes = sum(len(spikes) for spikes in simulator.spike_times)
        cv_isi = np.std(pop_rate) / np.mean(pop_rate) if np.mean(pop_rate) > 0 else 0
        peak_rate = np.max(pop_rate)

        met1, met2 = st.columns(2)
        with met1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{np.mean(pop_rate):.1f} Hz</h3>
                <p>Mean Population Rate</p>
            </div>
            """, unsafe_allow_html=True)
        with met2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{total_spikes:,}</h3>
                <p>Total Spikes</p>
            </div>
            """, unsafe_allow_html=True)

        met3, met4 = st.columns(2)
        with met3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{cv_isi:.3f}</h3>
                <p>Rate Variability (CV)</p>
            </div>
            """, unsafe_allow_html=True)
        with met4:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{peak_rate:.1f} Hz</h3>
                <p>Peak Rate</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("👈 Click 'Run Simulation' in the sidebar to generate population dynamics visualization")

        # Show preview plot using actual sidebar parameters
        st.markdown("### Preview: Expected Response Profile")
        t_preview = np.linspace(0, simulation_time, 1000)
        np.random.seed(42)  # Consistent preview
        rate_preview = base_firing_rate + noise_strength * 3 * np.sin(2 * np.pi * t_preview / 200) + np.random.normal(0, noise_strength * 1.5, len(t_preview))
        stim_mask = (t_preview >= stimulus_start) & (t_preview <= stimulus_start + stimulus_duration)
        rate_preview[stim_mask] += base_firing_rate * stimulus_strength
        rate_preview = np.maximum(rate_preview, 0)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t_preview, rate_preview, 'b-', alpha=0.8)
        ax.axvspan(stimulus_start, stimulus_start + stimulus_duration, alpha=0.3, color='red', label='Stimulus')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Population Rate (Hz)')
        ax.set_title(f'Preview: {N_neurons} neurons, {base_firing_rate} Hz base rate, stimulus ×{1+stimulus_strength:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        st.caption("This is an approximate preview. Run the simulation for accurate Gillespie-sampled dynamics.")

with tab2:
    st.markdown('<p class="section-header">🎯 Neural Raster Plot</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>🎯 <strong>What this shows:</strong> Each dot is a single spike from a single neuron. Each row = one neuron, x-axis = time.
    <strong>Vertical stripes</strong> mean neurons are firing together (synchrony).
    <strong>Scattered dots</strong> mean independent firing (asynchronous).
    Below: the inter-spike interval (ISI) distribution tells you how regular each neuron's firing is,
    and the rate distribution shows how rates vary across the population.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.simulation_done:
        simulator = st.session_state.simulator
        
        # Create raster plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spikes for subset of neurons for clarity
        n_show = min(100, N_neurons)
        colors = plt.cm.viridis(np.linspace(0, 1, n_show))
        
        for i in range(n_show):
            if simulator.spike_times[i]:
                spike_times = np.array(simulator.spike_times[i])
                ax.scatter(spike_times, [i] * len(spike_times), 
                          c=[colors[i]], s=1, alpha=0.7)
        
        # Highlight stimulus period
        ax.axvspan(stimulus_start, stimulus_start + stimulus_duration,
                  alpha=0.2, color='red', label='Stimulus Period')
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel(f'Neuron ID (showing {n_show}/{N_neurons})', fontsize=12)
        ax.set_title('Neural Spike Raster Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
        # Spike statistics
        st.markdown("### Spike Train Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Inter-spike interval distribution
            all_isis = []
            for spikes in simulator.spike_times[:50]:  # Analyze subset
                if len(spikes) > 1:
                    isis = np.diff(spikes)
                    all_isis.extend(isis)
            
            if all_isis:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(all_isis, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Inter-Spike Interval (ms)')
                ax.set_ylabel('Count')
                ax.set_title('ISI Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with col2:
            # Firing rate distribution across neurons
            individual_rates = []
            for spikes in simulator.spike_times:
                rate = len(spikes) / (simulation_time / 1000.0)  # Hz
                individual_rates.append(rate)
            
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(individual_rates, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax.axvline(np.mean(individual_rates), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(individual_rates):.1f} Hz')
            ax.set_xlabel('Firing Rate (Hz)')
            ax.set_ylabel('Number of Neurons')
            ax.set_title('Rate Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
    else:
        st.info("👈 Run simulation first to see raster plots and spike analysis")

with tab3:
    st.markdown('<p class="section-header">📈 Statistical Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight-box">
    <p>📈 <strong>What this shows:</strong> Quantitative analysis of the simulation output.
    The <strong>correlation matrix</strong> reveals which neurons tend to fire together (warm colors = correlated).
    The <strong>power spectrum</strong> reveals oscillatory patterns — peaks indicate rhythmic activity at specific frequencies.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.simulation_done:
        simulator = st.session_state.simulator
        pop_rate = st.session_state.pop_rate
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Population Statistics")
            
            # Basic statistics
            stats_data = {
                "Mean Rate": f"{np.mean(pop_rate):.2f} Hz",
                "Std Rate": f"{np.std(pop_rate):.2f} Hz",
                "Min Rate": f"{np.min(pop_rate):.2f} Hz",
                "Max Rate": f"{np.max(pop_rate):.2f} Hz",
                "Rate CV": f"{np.std(pop_rate)/np.mean(pop_rate):.3f}",
                "Total Spikes": f"{sum(len(spikes) for spikes in simulator.spike_times):,}"
            }
            
            for key, value in stats_data.items():
                st.metric(key, value)
        
        with col2:
            st.markdown("#### Correlation Analysis")
            
            # Calculate pairwise correlations for subset of neurons
            n_corr = min(20, N_neurons)
            correlation_matrix = np.corrcoef(simulator.spike_trains[:n_corr])
            
            fig, ax = plt.subplots(figsize=(5, 3.5))
            im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Spike Train Correlations')
            ax.set_xlabel('Neuron ID')
            ax.set_ylabel('Neuron ID')
            plt.colorbar(im, ax=ax, label='Correlation')
            st.pyplot(fig)
            
            mean_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            st.metric("Mean Pairwise Correlation", f"{mean_correlation:.3f}")
        
        # Spectral analysis
        st.markdown("#### Spectral Analysis")
        
        # Power spectrum of population rate
        from scipy.fft import fft, fftfreq
        
        # Remove DC component and apply window
        windowed_signal = pop_rate - np.mean(pop_rate)
        windowed_signal *= np.hanning(len(windowed_signal))
        
        # Compute FFT
        freqs = fftfreq(len(windowed_signal), d=simulator.dt/1000.0)  # Convert to seconds
        power_spectrum = np.abs(fft(windowed_signal))**2
        
        # Plot only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = power_spectrum[pos_mask]
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.loglog(freqs_pos, power_pos, 'b-', alpha=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Population Rate Power Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 100)
        st.pyplot(fig)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power_pos[freqs_pos < 50])  # Below 50 Hz
        dominant_freq = freqs_pos[dominant_freq_idx]
        st.metric("Dominant Frequency", f"{dominant_freq:.1f} Hz")
        
    else:
        st.info("👈 Run simulation first to see statistical analysis")

with tab4:
    st.markdown('<p class="section-header">🧮 Theoretical Background</p>', unsafe_allow_html=True)

    # Mathematical Framework Section
    st.markdown('<div class="subsection-header">📐 Mathematical Framework</div>', unsafe_allow_html=True)

    with st.expander("**1. Gillespie Algorithm (Stochastic Simulation Algorithm)**", expanded=True):
        st.markdown("""
The **Gillespie algorithm** (1977) provides *exact* stochastic simulation by treating neural spiking as a continuous-time Markov process.
        """)

        st.markdown("""
<div class="highlight-box">
<p>💡 Unlike discrete-time methods that approximate dynamics, Gillespie samples the <strong>exact timing</strong> of events with no discretization error.</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("**Algorithm Steps:**")
        st.markdown("""
<div class="algo-step">
    <div class="step-num">1</div>
    <div class="step-content"><strong>Compute total rate:</strong> Λ(t) = Σᵢ λᵢ(t), where λᵢ is neuron i's instantaneous firing rate</div>
</div>
<div class="algo-step">
    <div class="step-num">2</div>
    <div class="step-content"><strong>Sample waiting time:</strong> Δt ~ Exponential(Λ), giving P(Δt > τ) = exp(-Λτ)</div>
</div>
<div class="algo-step">
    <div class="step-num">3</div>
    <div class="step-content"><strong>Select firing neuron:</strong> P(neuron i fires) = λᵢ(t) / Λ(t)</div>
</div>
<div class="algo-step">
    <div class="step-num">4</div>
    <div class="step-content"><strong>Update state:</strong> Record spike, update refractory states, advance time by Δt</div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="key-point">
    <div class="key-point-icon">✓</div>
    <p><strong>Why exact?</strong> The exponential waiting time reflects the <em>memoryless</em> property of Poisson processes. Combined with proportional selection, this exactly samples from the underlying continuous-time dynamics.</p>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**2. Inhomogeneous Poisson Process with Refractory Period**", expanded=True):
        st.markdown("Each neuron generates spikes as a **time-varying Poisson process** with rate modulated by multiple factors:")

        st.latex(r"\lambda_i(t) = \lambda_0 \cdot S(t) \cdot R(t - t_{last}) \cdot \exp(\sigma \cdot \eta_i(t)) \cdot (1 + \alpha \cdot A(t))")

        st.markdown("**Parameter Definitions:**")
        st.markdown("""
<div class="def-item"><span class="def-term">λ₀</span><span class="def-desc">Base firing rate (Hz) — intrinsic excitability of the neuron</span></div>
<div class="def-item"><span class="def-term">S(t)</span><span class="def-desc">Stimulus gain function — external drive (1 + s during stimulation)</span></div>
<div class="def-item"><span class="def-term">R(Δt)</span><span class="def-desc">Refractory function — Heaviside step H(Δt - τᵣₑf), enforcing absolute refractory period</span></div>
<div class="def-item"><span class="def-term">ηᵢ(t)</span><span class="def-desc">Correlated Gaussian noise — synaptic variability</span></div>
<div class="def-item"><span class="def-term">A(t)</span><span class="def-desc">Recent population activity — recurrent network effects</span></div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
    <h5>⏱️ Refractory Period</h5>
    <p>After firing, a neuron enters an <strong>absolute refractory period</strong> (τᵣₑf ≈ 1-3 ms) during which it cannot fire again. This arises from Na⁺ channel inactivation.</p>
    <p><strong>Maximum firing rate:</strong> f_max = 1/τᵣₑf ≈ 300-1000 Hz</p>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**3. Correlated Noise via Cholesky Decomposition**", expanded=True):
        st.markdown("Real neural populations exhibit **correlated variability** — neurons don't fluctuate independently. We model this with multivariate Gaussian noise:")

        st.latex(r"\vec{\eta} = L \cdot \vec{z}, \quad \text{where } \vec{z} \sim \mathcal{N}(0, I) \text{ and } C = LL^T")

        st.markdown("**Correlation Structure:**")
        st.markdown("""
<div class="param-grid">
    <div class="param-card">
        <h6>Correlation Matrix</h6>
        <p>Cᵢⱼ = ρ for i ≠ j, and Cᵢᵢ = 1</p>
    </div>
    <div class="param-card">
        <h6>Cholesky Factor</h6>
        <p>L = chol(C) — lower triangular matrix</p>
    </div>
    <div class="param-card">
        <h6>Result</h6>
        <p>E[ηᵢηⱼ] = ρ for all neuron pairs</p>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
    <h5>🧬 Biological Origin</h5>
    <p>Correlated fluctuations arise from:</p>
    <ul>
        <li>Shared synaptic input from common presynaptic sources</li>
        <li>Common neuromodulatory signals (dopamine, acetylcholine)</li>
        <li>Recurrent connectivity within the network</li>
    </ul>
    <p><strong>Typical values:</strong> ρ ≈ 0.1-0.3 in cortex, reflecting functional coupling strength</p>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**4. Mean-Field Approximation**", expanded=True):
        st.markdown("For large populations (N → ∞), individual spike trains average out, and the **population rate** becomes deterministic:")

        st.latex(r"r(t) = \frac{1}{N} \sum_i \lambda_i(t) \rightarrow \mathbb{E}[\lambda(t)] \text{ as } N \rightarrow \infty")

        st.markdown("""
<div class="highlight-box">
<p>📊 The mean-field rate provides a baseline prediction that can be compared against stochastic simulations.</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("**Deviations from mean-field arise from:**")
        st.markdown("""
<div class="param-grid">
    <div class="param-card">
        <h6>Finite-Size Fluctuations</h6>
        <p>σᵣ ∝ 1/√N — smaller populations are noisier</p>
    </div>
    <div class="param-card">
        <h6>Correlated Noise</h6>
        <p>Shared variability doesn't average out across neurons</p>
    </div>
    <div class="param-card">
        <h6>Network Interactions</h6>
        <p>Feedback creates non-linear dynamics beyond mean-field</p>
    </div>
</div>
        """, unsafe_allow_html=True)

    # Biological Relevance Section
    st.markdown('<div class="subsection-header">🧠 Biological Relevance</div>', unsafe_allow_html=True)

    with st.expander("**Neural Population Coding**", expanded=True):
        st.markdown("""
<div class="highlight-box">
<p>🧠 <strong>Why populations?</strong> The brain doesn't rely on single neurons. Populations of 100-10,000+ neurons work together for reliable computation.</p>
</div>
        """, unsafe_allow_html=True)

        st.markdown("**Key advantages of population coding:**")
        st.markdown("""
<div class="param-grid">
    <div class="param-card">
        <h6>🎯 Noise Averaging</h6>
        <p>Individual neurons are unreliable (CV ≈ 1), but populations achieve precision through averaging</p>
    </div>
    <div class="param-card">
        <h6>📡 Information Multiplexing</h6>
        <p>Different aspects encoded in rate, timing, and correlations simultaneously</p>
    </div>
    <div class="param-card">
        <h6>🛡️ Robust Computation</h6>
        <p>Graceful degradation if individual neurons fail or become noisy</p>
    </div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
    <h5>📈 Population Rate Coding</h5>
    <p>The simplest neural code — stimulus intensity maps to mean firing rate. This simulator demonstrates this principle: <strong>stronger stimuli → higher population rate</strong> during the stimulus window.</p>
</div>
        """, unsafe_allow_html=True)

    with st.expander("**Experimental Techniques This Simulates**", expanded=True):
        st.markdown("""
| Technique | What It Measures | Simulator Analog |
|-----------|------------------|------------------|
| **Multi-electrode arrays** | Spike times from ~100 neurons | Raster plot, ISI distribution |
| **Calcium imaging** | Bulk fluorescence ∝ population activity | Population rate time series |
| **Local field potential (LFP)** | Summed synaptic currents | Power spectrum analysis |
| **Neuropixels probes** | 1000s of neurons simultaneously | Correlation matrices, population statistics |
        """)

    with st.expander("**Clinical Relevance**", expanded=True):
        st.markdown("Understanding population dynamics informs treatment of neurological disorders:")

        st.markdown("""
<div class="param-grid">
    <div class="param-card">
        <h6>⚡ Epilepsy</h6>
        <p>Pathological hypersynchrony — too much correlation, runaway excitation. Compare "Highly Excitable" regime.</p>
    </div>
    <div class="param-card">
        <h6>🧠 Parkinson's Disease</h6>
        <p>Abnormal beta oscillations (13-30 Hz) in basal ganglia. Our spectral analysis detects such rhythms.</p>
    </div>
    <div class="param-card">
        <h6>🔮 Schizophrenia</h6>
        <p>Disrupted gamma synchrony during working memory. Correlation structure matters for cognition.</p>
    </div>
    <div class="param-card">
        <h6>💤 Anesthesia</h6>
        <p>Loss of consciousness correlates with breakdown of complex dynamics → more regular, correlated activity.</p>
    </div>
</div>
        """, unsafe_allow_html=True)

    # Parameter Guide
    st.markdown('<div class="subsection-header">🎛️ Parameter Effects Guide</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="concept-card">
    <h5>🔥 Firing Rate Parameters</h5>
    <div class="def-item"><span class="def-term">Base Rate (λ₀)</span><span class="def-desc">Spontaneous activity. Cortical neurons: 1-20 Hz</span></div>
    <div class="def-item"><span class="def-term">Refractory (τᵣₑf)</span><span class="def-desc">Sets max rate = 1/τᵣₑf. 2ms → 500 Hz max</span></div>
    <div class="def-item"><span class="def-term">Stimulus Strength</span><span class="def-desc">Multiplicative gain. 1.0 = 2× rate during stimulus</span></div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="concept-card">
    <h5>🌊 Noise & Correlation</h5>
    <div class="def-item"><span class="def-term">Synaptic Noise (σ)</span><span class="def-desc">Log-normal rate modulation. Higher = more variability</span></div>
    <div class="def-item"><span class="def-term">Correlation (ρ)</span><span class="def-desc">Shared fluctuations. Cortex: ρ ≈ 0.1-0.3</span></div>
    <div class="def-item"><span class="def-term">Connectivity (α)</span><span class="def-desc">Recurrent feedback strength. Enables oscillations</span></div>
</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="concept-card">
    <h5>📊 What to Observe</h5>
    <div class="def-item"><span class="def-term">Population Rate</span><span class="def-desc">Should track stimulus. Look for latency and adaptation</span></div>
    <div class="def-item"><span class="def-term">Raster Plot</span><span class="def-desc">Vertical stripes = synchrony. Scattered = async</span></div>
    <div class="def-item"><span class="def-term">ISI Distribution</span><span class="def-desc">Exponential = Poisson. Peak at τᵣₑf = rate-limited</span></div>
    <div class="def-item"><span class="def-term">Correlations</span><span class="def-desc">Diagonal = independent. Off-diagonal = coupled</span></div>
    <div class="def-item"><span class="def-term">Power Spectrum</span><span class="def-desc">Peaks = oscillations. 1/f = scale-free</span></div>
</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="key-point">
    <div class="key-point-icon">🎯</div>
    <p><strong>Try:</strong> Increase correlation → vertical stripes in raster. Increase connectivity → oscillations in spectrum.</p>
</div>
        """, unsafe_allow_html=True)

    # Parameter Regimes
    st.markdown('<div class="subsection-header">🌐 Canonical Network States</div>', unsafe_allow_html=True)

    regime_col1, regime_col2, regime_col3 = st.columns(3)

    with regime_col1:
        st.markdown("""
<div class="state-card">
    <h4>🧘 Asynchronous Irregular</h4>
    <div class="state-subtitle">The "awake cortex" state</div>
    <div class="params">
        ρ &lt; 0.2 · σ &gt; 0.5 · α &lt; 0.1
    </div>
    <div class="props">
        <strong>Properties:</strong><br>
        • CV(ISI) ≈ 1 (Poisson-like)<br>
        • Low pairwise correlations<br>
        • Flat power spectrum<br>
        • Maximum information capacity
    </div>
</div>
        """, unsafe_allow_html=True)

    with regime_col2:
        st.markdown("""
<div class="state-card">
    <h4>🌊 Synchronous Oscillatory</h4>
    <div class="state-subtitle">Gamma rhythms, attention</div>
    <div class="params">
        ρ &gt; 0.5 · σ = 0.2-0.4 · α &gt; 0.15
    </div>
    <div class="props">
        <strong>Properties:</strong><br>
        • Periodic population bursts<br>
        • Clear spectral peak<br>
        • Coordinated processing<br>
        • Feature binding
    </div>
</div>
        """, unsafe_allow_html=True)

    with regime_col3:
        st.markdown("""
<div class="state-card">
    <h4>⚡ Highly Synchronous</h4>
    <div class="state-subtitle">Seizure-like, pathological</div>
    <div class="params">
        ρ &gt; 0.8 · λ₀ &gt; 30 Hz · α &gt; 0.3
    </div>
    <div class="props">
        <strong>Properties:</strong><br>
        • Massive synchrony<br>
        • Runaway excitation<br>
        • Lost information coding<br>
        • Clinical: epileptiform
    </div>
</div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>🧠 Neural Population Dynamics Simulator</strong></p>
    <p>
        <a href="https://github.com/kiranshay/neural-population-dynamics-simulator" target="_blank">GitHub</a> ·
        <a href="https://kiranshay.github.io" target="_blank">Portfolio</a> ·
        <a href="mailto:kiranshay123@gmail.com">Contact</a>
    </p>
    <p style="font-size: 0.85rem; color: #94a3b8;">Monte Carlo methods in computational neuroscience · Built by Kiran Shay</p>
</div>
""", unsafe_allow_html=True)

# Add session state initialization
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
