```python
"""
Data loading utilities for Allen Brain Observatory data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import h5py

try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
    ALLEN_SDK_AVAILABLE = True
except ImportError:
    ALLEN_SDK_AVAILABLE = False
    logging.warning("AllenSDK not available. Allen data loading will not work.")

logger = logging.getLogger(__name__)


class AllenDataLoader:
    """
    Load and preprocess Allen Brain Observatory data for simulation validation.
    
    Parameters:
    -----------
    cache_dir : str
        Directory to cache downloaded data
    """
    
    def __init__(self, cache_dir: str = "./allen_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        if ALLEN_SDK_AVAILABLE:
            self.cache = BrainObservatoryCache(manifest_file=str(self.cache_dir / 'manifest.json'))
        else:
            self.cache = None
            logger.warning("AllenSDK not available")
    
    def get_experiment_list(self, brain_area: str = 'VISp') -> pd.DataFrame:
        """
        Get list of available experiments.
        
        Parameters:
        -----------
        brain_area : str
            Brain area code (e.g., 'VISp', 'VISl', 'VISal')
            
        Returns:
        --------
        pd.DataFrame
            Experiment metadata
        """
        if not ALLEN_SDK_AVAILABLE:
            raise RuntimeError("AllenSDK not available")
        
        experiments = self.cache.get_ophys_experiments()
        experiments_df = pd.DataFrame(experiments)
        
        if brain_area:
            experiments_df = experiments_df[
                experiments_df['targeted_structure'] == brain_area
            ]
        
        logger.info(f"Found {len(experiments_df)} experiments in {brain_area}")
        return experiments_df
    
    def load_experiment_data(self, experiment_id: int) -> Dict[str, Any]:
        """
        Load data from a specific experiment.
        
        Parameters:
        -----------
        experiment_id : int
            Allen experiment ID
            
        Returns:
        --------
        Dict[str, Any]
            Experiment data including spike times, stimulus info, etc.
        """
        if not ALLEN_SDK_AVAILABLE:
            raise RuntimeError("AllenSDK not available")
        
        logger.info(f"Loading experiment {experiment_id}")
        
        # Get experiment data
        data_set = self.cache.get_ophys_experiment_data(experiment_id)
        
        # Extract fluorescence traces
        timestamps = data_set.get_ophys_timestamps()
        dff = data_set.get_dff_traces()
        
        # Extract cell information
        cell_specimens = data_set.get_cell_specimens()
        cell_ids = [cell['cell_specimen_id'] for cell in cell_specimens]
        
        # Get stimulus information
        stimulus_table = data_set.get_stimulus_table()
        
        # Convert to spike-like events using fluorescence threshold
        spike_times = self._extract_spike_times(dff[1], timestamps, threshold=2.0)
        
        return {
            'experiment_id': experiment_id,
            'timestamps': timestamps,
            'fluorescence_traces': dff[1],  # dF/F traces
            'cell_ids': cell_ids,
            'spike_times': spike_times,
            'stimulus_table': stimulus_table,
            'metadata': {
                'n_cells': len(cell_ids),
                'duration': timestamps[-1] - timestamps[0],
                'sampling_rate': 1.0 / np.mean(np.diff(timestamps))
            }
        }
    
    def _extract_spike_times(self, traces: np.ndarray, timestamps: np.ndarray,
                           threshold: float = 2.0) -> Dict[int, np.ndarray]:
        """
        Extract spike-like events from fluorescence traces.
        
        Parameters:
        -----------
        traces : np.ndarray
            Fluorescence traces (cells x time)
        timestamps : np.ndarray
            Time stamps for each frame
        threshold : float
            Threshold in standard deviations above mean
            
        Returns:
        --------
        Dict[int, np.ndarray]
            Spike times for each cell
        """
        spike_times = {}
        
        for cell_idx in range(traces.shape[0]):
            trace = traces[cell_idx]
            
            # Z-score the trace
            z_trace = (trace - np.mean(trace)) / np.std(trace)
            
            # Find threshold crossings
            above_threshold = z_trace > threshold
            crossings = np.diff(above_threshold.astype(int)) > 0
            
            # Get spike times