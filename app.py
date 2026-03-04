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
    initial_sidebar_state="expanded"
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

    /* Section Headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
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

# Sidebar controls
st.sidebar.markdown("## 🎛️ Simulation Parameters")

# Population parameters
st.sidebar.markdown("### Population Settings")
N_neurons = st.sidebar.slider("Number of Neurons", 50, 500, 200, 50)
simulation_time = st.sidebar.slider("Simulation Time (ms)", 100, 2000, 1000, 100)

# Neural parameters
st.sidebar.markdown("### Neural Properties")
base_firing_rate = st.sidebar.slider("Base Firing Rate (Hz)", 1.0, 50.0, 10.0, 1.0)
refractory_period = st.sidebar.slider("Refractory Period (ms)", 1.0, 10.0, 2.0, 0.5)
noise_strength = st.sidebar.slider("Synaptic Noise", 0.0, 2.0, 0.5, 0.1)

# Network parameters
st.sidebar.markdown("### Network Dynamics")
connectivity = st.sidebar.slider("Network Connectivity", 0.0, 0.5, 0.1, 0.05)
correlation_strength = st.sidebar.slider("Population Correlation", 0.0, 1.0, 0.3, 0.1)

# Stimulus parameters
st.sidebar.markdown("### External Stimulus")
stimulus_start = st.sidebar.slider("Stimulus Start (ms)", 100, 800, 300, 50)
stimulus_duration = st.sidebar.slider("Stimulus Duration (ms)", 50, 500, 200, 50)
stimulus_strength = st.sidebar.slider("Stimulus Strength", 0.0, 3.0, 1.0, 0.1)

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
            correlation_matrix = correlation * np.ones((self.N, self.N)) + (1-correlation) * np.eye(self.N)
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
                
                # Network coupling (simplified)
                recent_activity = np.sum([len([s for s in spikes if t_current - 10 < s <= t_current]) 
                                        for spikes in self.spike_times])
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
                        self.spike_trains[neuron_idx, spike_idx] = 1
                
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
    
    if st.session_state.simulation_done:
        simulator = st.session_state.simulator
        pop_rate = st.session_state.pop_rate
        
        # Create main plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>{np.mean(pop_rate):.1f} Hz</h3>
                <p>Mean Population Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_spikes = sum(len(spikes) for spikes in simulator.spike_times)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{total_spikes:,}</h3>
                <p>Total Spikes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_isi = np.std(pop_rate) / np.mean(pop_rate) if np.mean(pop_rate) > 0 else 0
            st.markdown(f"""
            <div class="metric-container">
                <h3>{cv_isi:.3f}</h3>
                <p>Rate Variability (CV)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            peak_rate = np.max(pop_rate)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{peak_rate:.1f} Hz</h3>
                <p>Peak Rate</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("👈 Click 'Run Simulation' in the sidebar to generate population dynamics visualization")
        
        # Show example plot
        st.markdown("### Example: What You'll See")
        t_example = np.linspace(0, 1000, 1000)
        rate_example = 10 + 5*np.sin(2*np.pi*t_example/200) + np.random.normal(0, 2, len(t_example))
        rate_example[300:500] += 15  # Stimulus response
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_example, rate_example, 'b-', alpha=0.8)
        ax.axvspan(300, 500, alpha=0.3, color='red', label='Stimulus')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Population Rate (Hz)')
        ax.set_title('Example: Neural Population Response to Stimulus')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

with tab2:
    st.markdown('<p class="section-header">🎯 Neural Raster Plot</p>', unsafe_allow_html=True)
    
    if st.session_state.simulation_done:
        simulator = st.session_state.simulator
        
        # Create raster plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
                fig, ax = plt.subplots(figsize=(6, 4))
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
            
            fig, ax = plt.subplots(figsize=(6, 4))
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
            
            fig, ax = plt.subplots(figsize=(6, 5))
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
        
        fig, ax = plt.subplots(figsize=(10, 4))
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
    st.markdown("### Mathematical Framework")

    with st.expander("**1. Gillespie Algorithm (Stochastic Simulation Algorithm)**", expanded=True):
        st.markdown("""
The **Gillespie algorithm** (1977) provides *exact* stochastic simulation by treating neural spiking as a continuous-time Markov process. Unlike discrete-time methods that approximate dynamics, Gillespie samples the exact timing of events.

**Algorithm Steps:**
1. **Compute total rate:** Λ(t) = Σᵢ λᵢ(t), where λᵢ is neuron i's instantaneous firing rate
2. **Sample waiting time:** Δt ~ Exponential(Λ), giving P(Δt > τ) = exp(-Λτ)
3. **Select firing neuron:** P(neuron i fires) = λᵢ(t) / Λ(t)
4. **Update state:** Record spike, update refractory states, advance time by Δt

**Why exact?** The exponential waiting time is the *memoryless* property of Poisson processes. Combined with proportional selection, this exactly samples from the underlying continuous-time dynamics without discretization error.
        """)

    with st.expander("**2. Inhomogeneous Poisson Process with Refractory Period**", expanded=True):
        st.markdown("""
Each neuron generates spikes as a **time-varying Poisson process** with rate modulated by multiple factors:
        """)
        st.latex(r"\lambda_i(t) = \lambda_0 \cdot S(t) \cdot R(t - t_{last}) \cdot \exp(\sigma \cdot \eta_i(t)) \cdot (1 + \alpha \cdot A(t))")
        st.markdown("""
**Where:**
- **λ₀** = Base firing rate (Hz) — intrinsic excitability
- **S(t)** = Stimulus gain function — external drive (1 + s during stimulation)
- **R(Δt)** = Refractory function — Heaviside step H(Δt - τᵣₑf), enforcing absolute refractory period
- **ηᵢ(t)** = Correlated Gaussian noise — synaptic variability
- **A(t)** = Recent population activity — recurrent network effects

**Refractory Period:** After firing, a neuron enters an *absolute refractory period* (τᵣₑf ≈ 1-3 ms) during which it cannot fire again. This arises from Na⁺ channel inactivation and sets the maximum firing rate: f_max = 1/τᵣₑf ≈ 300-1000 Hz.
        """)

    with st.expander("**3. Correlated Noise via Cholesky Decomposition**", expanded=True):
        st.markdown("""
Real neural populations exhibit **correlated variability** — neurons don't fluctuate independently. We model this with multivariate Gaussian noise:
        """)
        st.latex(r"\vec{\eta} = L \cdot \vec{z}, \quad \text{where } \vec{z} \sim \mathcal{N}(0, I) \text{ and } C = LL^T")
        st.markdown("""
**Correlation Structure:**
- **Correlation matrix:** Cᵢⱼ = ρ for i ≠ j, and Cᵢᵢ = 1
- **Cholesky factor:** L = chol(C) — lower triangular matrix
- **Result:** E[ηᵢηⱼ] = ρ for all neuron pairs

**Biological origin:** Correlated fluctuations arise from shared synaptic input, common neuromodulatory signals, and recurrent connectivity. Correlation magnitude (ρ ≈ 0.1-0.3 in cortex) reflects functional coupling strength.
        """)

    with st.expander("**4. Mean-Field Approximation**", expanded=True):
        st.markdown("""
For large populations (N → ∞), individual spike trains average out, and the **population rate** becomes deterministic:
        """)
        st.latex(r"r(t) = \frac{1}{N} \sum_i \lambda_i(t) \rightarrow \mathbb{E}[\lambda(t)] \text{ as } N \rightarrow \infty")
        st.markdown("""
The mean-field rate provides a baseline prediction. Deviations from mean-field arise from:
- **Finite-size fluctuations:** σᵣ ∝ 1/√N — smaller populations are noisier
- **Correlated noise:** Shared variability doesn't average out
- **Network interactions:** Feedback creates non-linear dynamics
        """)

    # Biological Relevance Section
    st.markdown("### Biological Relevance")

    with st.expander("**🧠 Neural Population Coding**", expanded=True):
        st.markdown("""
**Why populations?** The brain doesn't rely on single neurons. Populations of 100-10,000+ neurons work together to:
- **Average out noise:** Individual neurons are unreliable (CV ≈ 1), but populations achieve precision through averaging
- **Multiplex information:** Different aspects encoded in rate, timing, and correlations
- **Enable robust computation:** Graceful degradation if individual neurons fail

**Population rate coding:** The simplest code — stimulus intensity maps to mean firing rate. Our simulator shows this: stronger stimuli → higher population rate during the stimulus window.
        """)

    with st.expander("**🔬 Experimental Techniques This Simulates**", expanded=True):
        st.markdown("""
| Technique | What It Measures | Simulator Analog |
|-----------|------------------|------------------|
| **Multi-electrode arrays** | Spike times from ~100 neurons | Raster plot, ISI distribution |
| **Calcium imaging** | Bulk fluorescence ∝ population activity | Population rate time series |
| **Local field potential (LFP)** | Summed synaptic currents | Power spectrum analysis |
| **Neuropixels probes** | 1000s of neurons simultaneously | Correlation matrices, population statistics |
        """)

    with st.expander("**🏥 Clinical Relevance**", expanded=True):
        st.markdown("""
Understanding population dynamics informs treatment of neurological disorders:
- **Epilepsy:** Pathological hypersynchrony — too much correlation, runaway excitation. Compare "Highly Excitable" regime.
- **Parkinson's disease:** Abnormal beta oscillations (13-30 Hz) in basal ganglia. Our spectral analysis detects such rhythms.
- **Schizophrenia:** Disrupted gamma synchrony during working memory. Correlation structure matters for cognition.
- **Anesthesia:** Loss of consciousness correlates with breakdown of complex dynamics → more regular, correlated activity.
        """)

    # Parameter Guide
    st.markdown("### Parameter Effects Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔥 Firing Rate Parameters:**
        - **Base Rate (λ₀)**: Spontaneous activity level. Cortical neurons: 1-20 Hz. Higher = more spikes.
        - **Refractory Period (τᵣₑf)**: Sets max rate = 1/τᵣₑf. 2ms → 500 Hz max.
        - **Stimulus Strength**: Multiplicative gain. 1.0 = 2x rate during stimulus.

        **🌊 Noise & Correlation:**
        - **Synaptic Noise (σ)**: Log-normal rate modulation. Higher = more trial-to-trial variability.
        - **Population Correlation (ρ)**: Shared fluctuations. Cortex: ρ ≈ 0.1-0.3.
        - **Connectivity (α)**: Recurrent feedback strength. Enables oscillations.
        """)

    with col2:
        st.markdown("""
        **📊 What to Observe:**
        - **Population Rate**: Should track stimulus. Look for latency and adaptation.
        - **Raster Plot**: Vertical stripes = synchrony. Scattered = asynchronous.
        - **ISI Distribution**: Exponential = Poisson. Peak at τᵣₑf = rate-limited.
        - **Correlations**: Diagonal-heavy = independent. Off-diagonal = coupled.
        - **Power Spectrum**: Peaks = oscillations. 1/f slope = scale-free dynamics.

        **🎯 Try These Experiments:**
        - Increase correlation → watch vertical stripes emerge in raster
        - Increase connectivity → observe oscillations in power spectrum
        - Lower refractory → allow higher peak rates
        """)

    # Parameter Regimes
    st.markdown("### Canonical Network States")

    regime_col1, regime_col2, regime_col3 = st.columns(3)

    with regime_col1:
        st.markdown("""
        **🧘 Asynchronous Irregular (AI)**

        *The "awake cortex" state*

        - Correlation: < 0.2
        - Noise: > 0.5
        - Connectivity: < 0.1

        **Properties:**
        - CV(ISI) ≈ 1 (Poisson-like)
        - Low pairwise correlations
        - Flat power spectrum
        - Most information capacity
        """)

    with regime_col2:
        st.markdown("""
        **🌊 Synchronous Oscillatory**

        *Gamma rhythms, attention*

        - Correlation: > 0.5
        - Noise: 0.2-0.4
        - Connectivity: > 0.15

        **Properties:**
        - Periodic population bursts
        - Clear spectral peak
        - Coordinated processing
        - Feature binding
        """)

    with regime_col3:
        st.markdown("""
        **⚡ Highly Synchronous**

        *Seizure-like, pathological*

        - Correlation: > 0.8
        - Rate: > 30 Hz
        - Connectivity: > 0.3

        **Properties:**
        - Massive synchrony
        - Runaway excitation
        - Lost information coding
        - Clinical: epileptiform
        """)

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
