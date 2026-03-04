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
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .theory-box {
        background: #f8f9fa;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 0 1rem;
    }
    
    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #ddd;
        text-align: center;
        color: #666;
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
    st.markdown("### Population Firing Rate Dynamics")
    
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
    st.markdown("### Neural Raster Plot")
    
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
    st.markdown("### Statistical Analysis")
    
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
    st.markdown("### Theoretical Background")
    
    st.markdown("""
    <div class="theory-box">
    <h4>🧮 Mathematical Framework</h4>
    
    This simulator implements several key concepts from computational neuroscience:
    
    <h5>1. Gillespie Algorithm for Exact Stochastic Simulation</h5>
    The Gillespie algorithm provides exact stochastic simulation of reaction systems. For neural networks:
    
    - **Rate calculation**: λᵢ(t) = λ₀ᵢ × g(stimulus) × h(refractory) × noise
    - **Time step**: Δt ~ Exponential(Σλᵢ)
    - **Event selection**: P(neuron i fires) = λᵢ/Σλⱼ
    
    <h5>2. Poisson Process with Refractory Period</h5>
    Each neuron follows a modified Poisson process:
    
    - **Base rate**: λ₀ (spikes/sec)
    - **Refractory**: No spikes for τᵣₑf after each spike
    - **Effective rate**: λₑff(t) = λ₀ × [1 - H(τᵣₑf - (t - tₗₐₛₜ))]
    
    <h5>3. Correlated Noise Injection</h5>
    Population correlations via multivariate noise:
    
    - **Correlation matrix**: C = ρ×𝟙 + (1-ρ)×I
    - **Correlated noise**: 𝒏 = L×𝒛, where L=Chol(C), 𝒛~N(0,I)
    - **Rate modulation**: λᵢ(t) × exp(σ×nᵢ(t))
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="theory-box">
    <h4>🔬 Biological Relevance</h4>
    
    <h5>Neural Population Dynamics</h5>
    - **Spontaneous activity**: Baseline firing represents ongoing cortical computation
    - **Stimulus response**: External input drives coordinated population response  
    - **Network effects**: Recurrent connections create feedback and oscillations
    - **Variability**: Synaptic noise and ion channel stochasticity create trial-to-trial variability
    
    <h5>Experimental Parallels</h5>
    - **Calcium imaging**: Population rate ≈ bulk fluorescence signal
    - **Local field potential**: Spectral content reflects network oscillations
    - **Spike correlations**: Measure of functional connectivity
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive parameter effects
    st.markdown("### 🎛️ Parameter Effects Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔥 Firing Rate Parameters:**
        - **Base Rate**: Sets spontaneous activity level
        - **Stimulus Strength**: Multiplicative gain during stimulation
        - **Refractory Period**: Limits maximum firing rate
        
        **🌊 Noise & Correlation:**
        - **Synaptic Noise**: Increases rate variability
        - **Population Correlation**: Synchronizes neural activity
        - **Network Connectivity**: Creates feedback effects
        """)
    
    with col2:
        st.markdown("""
        **📊 Observables:**
        - **Population Rate**: Mean network activity
        - **Rate Variability**: CV of population dynamics
        - **Spike Correlations**: Pairwise neural coupling
        - **Power Spectrum**: Network oscillation frequencies
        
        **🎯 Biological Interpretation:**
        - High correlation → Synchronized states
        - Low noise → Regular, predictable activity  
        - Strong connectivity → Emergent oscillations
        """)
    
    # Show example parameter regimes
    st.markdown("### 📈 Example Parameter Regimes")
    
    regime_col1, regime_col2, regime_col3 = st.columns(3)
    
    with regime_col1:
        st.markdown("""
        **🧘 Asynchronous Irregular**
        - Low correlation (< 0.2)
        - High noise (> 0.8)
        - Weak connectivity (< 0.1)
        - *→ Realistic cortical state*
        """)
    
    with regime_col2:
        st.markdown("""
        **🌊 Synchronized Oscillations**
        - High correlation (> 0.6)
        - Moderate noise (~0.3)
        - Strong connectivity (> 0.2)
        - *→ Gamma/beta rhythms*
        """)
    
    with regime_col3:
        st.markdown("""
        **⚡ Highly Excitable**
        - High base rate (> 30 Hz)
        - Low refractory (< 2 ms)
        - High stimulus gain (> 2.0)
        - *→ Seizure-like activity*
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>🧠 Neural Population Dynamics Simulator | Built with Streamlit</p>
    <p>
        <a href="https://github.com/your-username/neural-population-dynamics" target="_blank">🔗 GitHub Repository</a> | 
        <a href="https://your-portfolio.com" target="_blank">🌐 Portfolio</a> |
        <a href="mailto:your.email@domain.com">📧 Contact</a>
    </p>
    <p><em>Demonstrating advanced Monte Carlo methods in computational neuroscience</em></p>
</div>
""", unsafe_allow_html=True)

# Add session state initialization
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
