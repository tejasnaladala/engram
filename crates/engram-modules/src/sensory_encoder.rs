use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, SpikeEvent, SimTime,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Sensory Encoder: converts continuous observation vectors into spike trains
/// using population rate coding.
///
/// For each input dimension, a population of neurons encodes the value.
/// Each neuron has a "preferred value" and fires proportionally to how
/// close the actual input is to its preference (Gaussian tuning curve).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryEncoder {
    pub population: NeuronPopulation,
    pub input_dims: usize,
    pub neurons_per_dim: usize,
    /// Preferred values for each neuron (length = input_dims * neurons_per_dim)
    pub preferred_values: Vec<f64>,
    /// Width of Gaussian tuning curve
    pub sigma: f64,
    /// Maximum firing rate (probability per ms)
    pub max_rate: f64,
    /// Current observation
    observation: Vec<f64>,
    /// Seed for RNG reconstruction
    seed: u64,
    /// RNG for spike generation
    #[serde(skip, default = "default_rng")]
    rng: ChaCha8Rng,
    /// Recent spike count for metrics
    recent_spike_count: u32,
}

fn default_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

impl SensoryEncoder {
    pub fn new(input_dims: usize, neurons_per_dim: usize, seed: u64) -> Self {
        let total = input_dims * neurons_per_dim;
        let params = LIFParams {
            tau_m: 10.0,     // fast response
            v_rest: -65.0,
            v_threshold: -55.0,
            v_reset: -70.0,
            r_membrane: 15.0,
            refractory_ms: 1.0, // short refractory
        };

        // Create preferred values uniformly spaced in [0, 1] for each dimension
        let mut preferred = Vec::with_capacity(total);
        for _dim in 0..input_dims {
            for i in 0..neurons_per_dim {
                preferred.push(i as f64 / (neurons_per_dim - 1).max(1) as f64);
            }
        }

        Self {
            population: NeuronPopulation::new(total, params),
            input_dims,
            neurons_per_dim,
            preferred_values: preferred,
            sigma: 0.15,
            max_rate: 0.3, // 30% chance per ms at peak
            observation: vec![0.0; input_dims],
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            recent_spike_count: 0,
        }
    }

    /// Set the current observation (values in [0, 1])
    pub fn set_observation(&mut self, obs: &[f64]) {
        let len = obs.len().min(self.input_dims);
        self.observation[..len].copy_from_slice(&obs[..len]);
    }
}

impl BrainModule for SensoryEncoder {
    fn id(&self) -> ModuleId {
        ModuleId::Sensory
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, _incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();

        // Generate input currents based on observation and tuning curves
        for dim in 0..self.input_dims {
            let obs_val = self.observation[dim];
            for i in 0..self.neurons_per_dim {
                let neuron_idx = dim * self.neurons_per_dim + i;
                let pref = self.preferred_values[neuron_idx];
                let diff = obs_val - pref;
                let activation = (-diff * diff / (2.0 * self.sigma * self.sigma)).exp();
                let rate = self.max_rate * activation;

                // Poisson spike generation
                if self.rng.random::<f64>() < rate * dt / 1.0 {
                    self.population.deliver_input(neuron_idx as u32, 20.0);
                }
            }
        }

        // Step all neurons
        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        for nid in spiked {
            spikes.push(SpikeEvent {
                source_module: ModuleId::Sensory,
                neuron_id: nid,
                timestamp: sim_time,
                strength: 1.0,
            });
        }

        spikes
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.population.len() as u32;
        ModuleSnapshot {
            module_id: ModuleId::Sensory,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0, // per second estimate
            activity_level: (self.recent_spike_count as f32 / n as f32).min(1.0),
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.observation.fill(0.0);
        self.recent_spike_count = 0;
    }

    fn neuron_count(&self) -> u32 {
        self.population.len() as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<SensoryEncoder>(data) {
            *self = restored;
        }
    }
}
