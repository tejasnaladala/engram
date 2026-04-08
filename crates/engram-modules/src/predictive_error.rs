use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, SpikeEvent, SimTime,
};
use serde::{Deserialize, Serialize};

/// Predictive Error Module: computes mismatch between predicted and actual
/// sensory input, generating surprise signals that modulate learning.
///
/// Maintains two trace vectors (predicted and actual) that accumulate
/// spike-weighted evidence. The L2 norm of their difference is the
/// prediction error, which feeds back to modulate STDP learning rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveError {
    pub population: NeuronPopulation,
    /// Accumulated actual sensory pattern (from sensory encoder)
    actual_trace: Vec<f64>,
    /// Accumulated predicted pattern (from associative memory)
    predicted_trace: Vec<f64>,
    /// Trace decay time constant (ms)
    trace_tau: f64,
    /// Current prediction error (scalar)
    pub error: f64,
    /// Error history for smoothing
    error_history: Vec<f64>,
    /// Number of error neurons
    num_neurons: usize,
    recent_spike_count: u32,
}

impl PredictiveError {
    pub fn new(num_neurons: usize, trace_size: usize) -> Self {
        let params = LIFParams {
            tau_m: 15.0,
            v_rest: -65.0,
            v_threshold: -50.0, // more excitable — error should be responsive
            v_reset: -70.0,
            r_membrane: 12.0,
            refractory_ms: 1.5,
        };

        Self {
            population: NeuronPopulation::new(num_neurons, params),
            actual_trace: vec![0.0; trace_size],
            predicted_trace: vec![0.0; trace_size],
            trace_tau: 50.0,
            error: 0.0,
            error_history: Vec::with_capacity(100),
            num_neurons,
            recent_spike_count: 0,
        }
    }

    /// Feed actual sensory spikes into the actual trace
    pub fn receive_actual(&mut self, spikes: &[SpikeEvent]) {
        for spike in spikes {
            let idx = spike.neuron_id as usize % self.actual_trace.len();
            self.actual_trace[idx] += spike.strength as f64;
        }
    }

    /// Feed predicted spikes (from associative memory) into the predicted trace
    pub fn receive_predicted(&mut self, spikes: &[SpikeEvent]) {
        for spike in spikes {
            let idx = spike.neuron_id as usize % self.predicted_trace.len();
            self.predicted_trace[idx] += spike.strength as f64;
        }
    }

    /// Get the current prediction error as a learning rate modifier.
    /// High error → high learning rate. Low error → low learning rate.
    pub fn learning_rate_modifier(&self) -> f64 {
        // Sigmoid-like mapping: error of ~0.5 gives modifier of ~1.0
        let clamped = self.error.clamp(0.0, 2.0);
        0.2 + 1.6 * clamped / (1.0 + clamped)
    }
}

impl BrainModule for PredictiveError {
    fn id(&self) -> ModuleId {
        ModuleId::PredictiveError
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        // Route incoming spikes based on source
        let actual: Vec<_> = incoming
            .iter()
            .filter(|s| s.source_module == ModuleId::Sensory)
            .cloned()
            .collect();
        let predicted: Vec<_> = incoming
            .iter()
            .filter(|s| s.source_module == ModuleId::AssociativeMemory)
            .cloned()
            .collect();

        self.receive_actual(&actual);
        self.receive_predicted(&predicted);

        // Decay traces
        let decay = (-dt / self.trace_tau).exp();
        for t in &mut self.actual_trace {
            *t *= decay;
        }
        for t in &mut self.predicted_trace {
            *t *= decay;
        }

        // Compute L2 error between actual and predicted
        let mut sum_sq = 0.0;
        for (a, p) in self.actual_trace.iter().zip(self.predicted_trace.iter()) {
            let diff = a - p;
            sum_sq += diff * diff;
        }
        self.error = (sum_sq / self.actual_trace.len() as f64).sqrt();

        // Record error history
        if self.error_history.len() >= 100 {
            self.error_history.remove(0);
        }
        self.error_history.push(self.error);

        // Drive error neurons proportionally to error magnitude
        let error_current = self.error * 10.0;
        for i in 0..self.num_neurons {
            // Scale current by neuron index to create a gradient of sensitivity
            let scale = (i as f64 + 1.0) / self.num_neurons as f64;
            if error_current * scale > 0.5 {
                self.population.deliver_input(i as u32, error_current * scale);
            }
        }

        // Step neurons
        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        spiked
            .into_iter()
            .map(|nid| SpikeEvent {
                source_module: ModuleId::PredictiveError,
                neuron_id: nid,
                timestamp: sim_time,
                strength: self.error as f32,
            })
            .collect()
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.num_neurons as u32;
        ModuleSnapshot {
            module_id: ModuleId::PredictiveError,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0,
            activity_level: (self.error as f32).clamp(0.0, 1.0),
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.actual_trace.fill(0.0);
        self.predicted_trace.fill(0.0);
        self.error = 0.0;
        self.error_history.clear();
        self.recent_spike_count = 0;
    }

    fn neuron_count(&self) -> u32 {
        self.num_neurons as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<PredictiveError>(data) {
            *self = restored;
        }
    }
}
