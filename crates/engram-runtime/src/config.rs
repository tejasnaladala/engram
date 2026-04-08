use serde::{Deserialize, Serialize};

/// Configuration for the Engram cognitive runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Simulation timestep (ms)
    pub dt: f64,
    /// Number of input dimensions
    pub input_dims: usize,
    /// Neurons per input dimension in sensory encoder
    pub sensory_neurons_per_dim: usize,
    /// Number of discrete actions
    pub num_actions: usize,
    /// Neurons per action pool
    pub neurons_per_action: usize,
    /// Number of associative memory neurons
    pub assoc_neurons: usize,
    /// Number of SDM hard locations
    pub sdm_locations: usize,
    /// SDM data width
    pub sdm_data_width: usize,
    /// Number of predictive error neurons
    pub pred_error_neurons: usize,
    /// Number of episodic memory neurons
    pub episodic_neurons: usize,
    /// Number of safety kernel neurons
    pub safety_neurons: usize,
    /// Maximum stored episodes
    pub max_episodes: usize,
    /// Synapse density for inter-module connections
    pub synapse_density: f64,
    /// Initial synapse weight maximum
    pub w_init_max: f32,
    /// Random seed for deterministic reproduction
    pub seed: u64,
    /// Dashboard frame rate (fps)
    pub dashboard_fps: u32,
    /// WebSocket port
    pub ws_port: u16,
    /// Enable safety kernel
    pub safety_enabled: bool,
    /// Enable episodic replay
    pub replay_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            dt: 1.0,
            input_dims: 8,
            sensory_neurons_per_dim: 16,
            num_actions: 4,
            neurons_per_action: 32,
            assoc_neurons: 256,
            sdm_locations: 1000,
            sdm_data_width: 128,
            pred_error_neurons: 64,
            episodic_neurons: 64,
            safety_neurons: 32,
            max_episodes: 100,
            synapse_density: 0.3,
            w_init_max: 0.5,
            seed: 42,
            dashboard_fps: 30,
            ws_port: 9000,
            safety_enabled: true,
            replay_enabled: true,
        }
    }
}
