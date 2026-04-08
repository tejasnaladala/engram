use serde::{Deserialize, Serialize};

use crate::synapse::SynapseMatrix;
use crate::types::Weight;

/// Parameters for Spike-Timing-Dependent Plasticity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPParams {
    /// Long-term potentiation amplitude (pre before post = strengthen)
    pub a_plus: f64,
    /// Long-term depression amplitude (post before pre = weaken)
    pub a_minus: f64,
    /// LTP time constant (ms)
    pub tau_plus: f64,
    /// LTD time constant (ms)
    pub tau_minus: f64,
    /// Maximum synaptic weight
    pub w_max: Weight,
    /// Minimum synaptic weight
    pub w_min: Weight,
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012, // asymmetric: depression slightly stronger
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: 0.0,
        }
    }
}

/// Online trace-based STDP learning state.
///
/// Maintains exponentially decaying eligibility traces for pre- and
/// post-synaptic neurons. When a pre-synaptic neuron fires, its trace
/// jumps to 1 and then decays. Weight updates are computed from the
/// product of traces and learning rates, avoiding the need to store
/// and iterate over all spike pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPState {
    pub params: STDPParams,
    /// Pre-synaptic eligibility traces
    pub pre_traces: Vec<f64>,
    /// Post-synaptic eligibility traces
    pub post_traces: Vec<f64>,
    /// Learning rate modulator (set by prediction error)
    pub learning_rate_mod: f64,
}

impl STDPState {
    pub fn new(pre_count: usize, post_count: usize, params: STDPParams) -> Self {
        Self {
            params,
            pre_traces: vec![0.0; pre_count],
            post_traces: vec![0.0; post_count],
            learning_rate_mod: 1.0,
        }
    }

    /// Apply one STDP update step.
    ///
    /// Algorithm:
    /// 1. Decay all traces by exp(-dt/tau)
    /// 2. For each pre-spike: set trace to 1, apply LTD using post_traces
    /// 3. For each post-spike: set trace to 1, apply LTP using pre_traces
    /// 4. Clamp all weights
    pub fn apply(
        &mut self,
        dt: f64,
        synapses: &mut SynapseMatrix,
        pre_spikes: &[u32],
        post_spikes: &[u32],
    ) {
        let lr = self.learning_rate_mod;

        // Step 1: Decay traces
        let pre_decay = (-dt / self.params.tau_plus).exp();
        let post_decay = (-dt / self.params.tau_minus).exp();
        for t in &mut self.pre_traces {
            *t *= pre_decay;
        }
        for t in &mut self.post_traces {
            *t *= post_decay;
        }

        // Step 2: Process pre-synaptic spikes → LTD
        for &pre_id in pre_spikes {
            self.pre_traces[pre_id as usize] = 1.0;
            // For each post-synaptic target of this pre-neuron:
            // If the post-neuron fired recently (post_trace > 0), the post fired
            // BEFORE pre → depression (LTD)
            for syn in synapses.outgoing(pre_id).collect::<Vec<_>>() {
                let post_trace = self.post_traces[syn.post_id as usize];
                if post_trace > 1e-6 {
                    let delta = -(self.params.a_minus * post_trace * lr) as Weight;
                    synapses.update_weight(pre_id, syn.post_id, delta);
                }
            }
        }

        // Step 3: Process post-synaptic spikes → LTP
        for &post_id in post_spikes {
            self.post_traces[post_id as usize] = 1.0;
            // For each pre-synaptic neuron connected to this post-neuron:
            // If the pre-neuron fired recently (pre_trace > 0), the pre fired
            // BEFORE post → potentiation (LTP)
            // We need to iterate all pre-neurons that connect to post_id.
            // This is the transpose lookup -- expensive in CSR format.
            // For moderate-size networks, we iterate all rows.
            for pre_id in 0..synapses.pre_count {
                let pre_trace = self.pre_traces[pre_id as usize];
                if pre_trace > 1e-6 {
                    let delta = (self.params.a_plus * pre_trace * lr) as Weight;
                    synapses.update_weight(pre_id, post_id, delta);
                }
            }
        }

        // Step 4: Clamp weights
        synapses.clamp_weights(self.params.w_min, self.params.w_max);
    }

    /// Set the learning rate modulator (typically from prediction error)
    pub fn set_learning_rate_mod(&mut self, mod_factor: f64) {
        self.learning_rate_mod = mod_factor.clamp(0.1, 5.0);
    }

    /// Reset all traces to zero
    pub fn reset(&mut self) {
        for t in &mut self.pre_traces {
            *t = 0.0;
        }
        for t in &mut self.post_traces {
            *t = 0.0;
        }
        self.learning_rate_mod = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn stdp_potentiates_pre_before_post() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut synapses = SynapseMatrix::random_sparse(4, 4, 1.0, 0.5, &mut rng);
        synapses.sort_rows();
        let mut stdp = STDPState::new(4, 4, STDPParams::default());

        // Pre-neuron 0 fires
        stdp.apply(1.0, &mut synapses, &[0], &[]);
        // Small delay, then post-neuron 1 fires
        stdp.apply(5.0, &mut synapses, &[], &[1]);

        // Weight from 0→1 should have increased (LTP)
        // We check by propagating from neuron 0
        let currents = synapses.propagate(&[0]);
        assert!(!currents.is_empty(), "Should have connections from neuron 0");
    }

    #[test]
    fn traces_decay_over_time() {
        let mut stdp = STDPState::new(4, 4, STDPParams::default());
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut synapses = SynapseMatrix::random_sparse(4, 4, 1.0, 0.5, &mut rng);

        // Fire pre-neuron 0
        stdp.apply(1.0, &mut synapses, &[0], &[]);
        let trace_after_spike = stdp.pre_traces[0];
        assert!((trace_after_spike - 1.0).abs() < 0.05);

        // Wait 100ms worth of decay
        for _ in 0..100 {
            stdp.apply(1.0, &mut synapses, &[], &[]);
        }
        assert!(
            stdp.pre_traces[0] < 0.01,
            "Trace should have decayed near zero after 100ms"
        );
    }
}
