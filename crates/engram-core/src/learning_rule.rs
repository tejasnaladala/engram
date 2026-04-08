use serde::{Deserialize, Serialize};
use crate::synapse::SynapseMatrix;
use crate::types::Weight;

/// Global neuromodulatory signals that influence learning across the network.
/// Inspired by dopamine (reward), acetylcholine (attention), norepinephrine
/// (arousal), and serotonin (inhibition/patience).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Neuromodulators {
    /// Reward prediction error (dopamine-like). Positive = better than expected.
    pub reward_signal: f64,
    /// Surprise / novelty signal (acetylcholine-like). High = unexpected input.
    pub surprise_signal: f64,
    /// Arousal / urgency (norepinephrine-like). High = heightened responsiveness.
    pub arousal_signal: f64,
    /// Inhibition / patience (serotonin-like). High = reduced exploration.
    pub inhibition_signal: f64,
}

impl Neuromodulators {
    /// Compute the composite modulatory signal for learning.
    /// This is the "third factor" in three-factor learning rules.
    pub fn learning_modulation(&self) -> f64 {
        // Primary driver is reward signal; surprise amplifies learning;
        // arousal scales overall magnitude; inhibition dampens
        let base = self.reward_signal;
        let surprise_boost = 1.0 + self.surprise_signal.abs() * 0.5;
        let arousal_scale = 0.5 + self.arousal_signal * 0.5;
        let inhibition_damp = 1.0 - self.inhibition_signal * 0.3;
        base * surprise_boost * arousal_scale * inhibition_damp
    }

    /// Update signals from environment feedback
    pub fn update(&mut self, reward: f64, prediction_error: f64, reward_baseline: &mut f64) {
        // Reward prediction error (RPE) = actual reward - expected reward
        let rpe = reward - *reward_baseline;
        *reward_baseline = *reward_baseline * 0.99 + reward * 0.01; // exponential moving avg

        self.reward_signal = rpe;
        self.surprise_signal = prediction_error;
        // Arousal increases with absolute RPE (something happened!)
        self.arousal_signal = (self.arousal_signal * 0.95 + rpe.abs() * 0.3).clamp(0.0, 1.0);
        // Inhibition increases when things are going well (less exploration needed)
        self.inhibition_signal = if rpe > 0.0 {
            (self.inhibition_signal + 0.02).clamp(0.0, 0.8)
        } else {
            (self.inhibition_signal - 0.05).clamp(0.0, 0.8)
        };
    }

    /// Reset all signals
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The LearningRule trait — the "autograd" equivalent for adaptive systems.
/// Every learning rule implements this interface. Rules are applied locally
/// to individual synapse matrices, with access to neuromodulatory signals.
pub trait LearningRule: Send {
    /// Apply one learning step.
    ///
    /// Arguments:
    /// - dt: timestep in ms
    /// - synapses: the synapse matrix to update
    /// - pre_spikes: IDs of pre-synaptic neurons that fired this tick
    /// - post_spikes: IDs of post-synaptic neurons that fired this tick
    /// - modulators: global neuromodulatory signals
    fn apply(
        &mut self,
        dt: f64,
        synapses: &mut SynapseMatrix,
        pre_spikes: &[u32],
        post_spikes: &[u32],
        modulators: &Neuromodulators,
    );

    /// Reset internal state (traces, etc.)
    fn reset(&mut self);

    /// Name of this learning rule (for logging/dashboard)
    fn name(&self) -> &'static str;
}

/// Three-factor STDP with eligibility traces.
///
/// This is the core learning mechanism that enables credit assignment
/// in spiking networks. The weight update formula:
///
///   dw_ij = eta * e_ij * M(t)
///
/// where:
///   e_ij = eligibility trace (accumulated from STDP, decays with tau_e)
///   M(t) = modulatory signal (reward prediction error from neuromodulators)
///
/// The eligibility trace bridges the temporal gap between spike correlations
/// and delayed reward, enabling the system to learn from experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeFactorSTDP {
    // STDP parameters
    pub a_plus: f64,
    pub a_minus: f64,
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub w_max: Weight,
    pub w_min: Weight,

    // Eligibility trace parameters
    pub tau_eligibility: f64, // decay time constant for eligibility (ms)
    pub eta: f64,             // base learning rate

    // Internal state
    pre_traces: Vec<f64>,
    post_traces: Vec<f64>,
    /// Per-synapse eligibility traces (sparse — only stored for active synapses)
    eligibility: Vec<f64>,
}

impl ThreeFactorSTDP {
    pub fn new(pre_count: usize, post_count: usize, synapse_count: usize) -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: 0.0,
            tau_eligibility: 1000.0, // 1 second — long enough for delayed reward
            eta: 0.005,
            pre_traces: vec![0.0; pre_count],
            post_traces: vec![0.0; post_count],
            eligibility: vec![0.0; synapse_count],
        }
    }

    /// Create with custom parameters
    pub fn with_params(
        pre_count: usize,
        post_count: usize,
        synapse_count: usize,
        tau_eligibility: f64,
        eta: f64,
    ) -> Self {
        let mut s = Self::new(pre_count, post_count, synapse_count);
        s.tau_eligibility = tau_eligibility;
        s.eta = eta;
        s
    }
}

impl LearningRule for ThreeFactorSTDP {
    fn apply(
        &mut self,
        dt: f64,
        synapses: &mut SynapseMatrix,
        pre_spikes: &[u32],
        post_spikes: &[u32],
        modulators: &Neuromodulators,
    ) {
        // === Phase 1: Decay all traces ===
        let pre_decay = (-dt / self.tau_plus).exp();
        let post_decay = (-dt / self.tau_minus).exp();
        let elig_decay = (-dt / self.tau_eligibility).exp();

        for t in &mut self.pre_traces { *t *= pre_decay; }
        for t in &mut self.post_traces { *t *= post_decay; }
        for e in &mut self.eligibility { *e *= elig_decay; }

        // === Phase 2: Compute STDP and accumulate into eligibility traces ===

        // Pre-spikes: update pre traces, compute LTD contribution to eligibility
        for &pre_id in pre_spikes {
            self.pre_traces[pre_id as usize] = 1.0;

            for syn in synapses.outgoing(pre_id).collect::<Vec<_>>() {
                let post_trace = self.post_traces[syn.post_id as usize];
                if post_trace > 1e-6 {
                    // Post fired before pre → LTD → negative eligibility
                    let start = synapses.row_ptr[pre_id as usize] as usize;
                    let end = synapses.row_ptr[(pre_id + 1) as usize] as usize;
                    for idx in start..end {
                        if synapses.col_idx[idx] == syn.post_id {
                            self.eligibility[idx] -= self.a_minus * post_trace;
                            break;
                        }
                    }
                }
            }
        }

        // Post-spikes: update post traces, compute LTP contribution to eligibility
        for &post_id in post_spikes {
            self.post_traces[post_id as usize] = 1.0;

            // Iterate all pre-neurons to find connections to this post-neuron
            for pre_id in 0..synapses.pre_count {
                let pre_trace = self.pre_traces[pre_id as usize];
                if pre_trace > 1e-6 {
                    // Pre fired before post → LTP → positive eligibility
                    let start = synapses.row_ptr[pre_id as usize] as usize;
                    let end = synapses.row_ptr[(pre_id + 1) as usize] as usize;
                    for idx in start..end {
                        if synapses.col_idx[idx] == post_id {
                            self.eligibility[idx] += self.a_plus * pre_trace;
                            break;
                        }
                    }
                }
            }
        }

        // === Phase 3: Apply weight updates using eligibility * modulation ===
        let modulation = modulators.learning_modulation();
        if modulation.abs() > 1e-6 {
            for idx in 0..synapses.values.len() {
                let delta = (self.eta * self.eligibility[idx] * modulation) as Weight;
                if delta.abs() > 1e-8 {
                    synapses.values[idx] += delta;
                }
            }
            synapses.clamp_weights(self.w_min, self.w_max);
        }
    }

    fn reset(&mut self) {
        self.pre_traces.fill(0.0);
        self.post_traces.fill(0.0);
        self.eligibility.fill(0.0);
    }

    fn name(&self) -> &'static str {
        "ThreeFactorSTDP"
    }
}

/// Simple Hebbian learning rule — "fire together, wire together."
/// No eligibility traces, no modulation. Pure co-activation strengthening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianRule {
    pub eta: f64,
    pub w_max: Weight,
    pub w_min: Weight,
    pre_traces: Vec<f64>,
    post_traces: Vec<f64>,
    pub tau: f64,
}

impl HebbianRule {
    pub fn new(pre_count: usize, post_count: usize) -> Self {
        Self {
            eta: 0.001,
            w_max: 1.0,
            w_min: 0.0,
            pre_traces: vec![0.0; pre_count],
            post_traces: vec![0.0; post_count],
            tau: 20.0,
        }
    }
}

impl LearningRule for HebbianRule {
    fn apply(
        &mut self,
        dt: f64,
        synapses: &mut SynapseMatrix,
        pre_spikes: &[u32],
        post_spikes: &[u32],
        _modulators: &Neuromodulators,
    ) {
        let decay = (-dt / self.tau).exp();
        for t in &mut self.pre_traces { *t *= decay; }
        for t in &mut self.post_traces { *t *= decay; }

        for &pid in pre_spikes { self.pre_traces[pid as usize] = 1.0; }
        for &pid in post_spikes { self.post_traces[pid as usize] = 1.0; }

        // Strengthen connections where both pre and post are active
        for &post_id in post_spikes {
            for pre_id in 0..synapses.pre_count {
                if self.pre_traces[pre_id as usize] > 0.1 {
                    let delta = (self.eta * self.pre_traces[pre_id as usize]) as Weight;
                    synapses.update_weight(pre_id, post_id, delta);
                }
            }
        }
        synapses.clamp_weights(self.w_min, self.w_max);
    }

    fn reset(&mut self) {
        self.pre_traces.fill(0.0);
        self.post_traces.fill(0.0);
    }

    fn name(&self) -> &'static str { "Hebbian" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn three_factor_eligibility_accumulates() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut syn = SynapseMatrix::random_sparse(4, 4, 1.0, 0.5, &mut rng);
        syn.sort_rows();
        let mut rule = ThreeFactorSTDP::new(4, 4, syn.nnz());
        let mods = Neuromodulators::default();

        // Pre fires, then post fires → eligibility should be positive
        rule.apply(1.0, &mut syn, &[0], &[], &mods);
        rule.apply(5.0, &mut syn, &[], &[1], &mods);

        // Some eligibility traces should be non-zero
        let has_nonzero = rule.eligibility.iter().any(|&e| e.abs() > 1e-6);
        assert!(has_nonzero, "Eligibility traces should accumulate from spike pairs");
    }

    #[test]
    fn three_factor_reward_drives_weight_change() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut syn = SynapseMatrix::random_sparse(4, 4, 1.0, 0.5, &mut rng);
        syn.sort_rows();
        let initial_weights: Vec<f32> = syn.values.clone();
        let mut rule = ThreeFactorSTDP::new(4, 4, syn.nnz());

        // Build up eligibility
        let no_mod = Neuromodulators::default();
        rule.apply(1.0, &mut syn, &[0], &[], &no_mod);
        rule.apply(5.0, &mut syn, &[], &[1, 2, 3], &no_mod);

        // Without reward, weights shouldn't change much
        let pre_reward_weights: Vec<f32> = syn.values.clone();

        // Now deliver reward signal
        let mut mods = Neuromodulators::default();
        mods.reward_signal = 1.0; // strong positive reward
        rule.apply(1.0, &mut syn, &[], &[], &mods);

        // Weights should change in the direction of eligibility
        let changed = syn.values.iter().zip(pre_reward_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "Reward should drive weight changes via eligibility traces");
    }

    #[test]
    fn neuromodulators_update_from_reward() {
        let mut mods = Neuromodulators::default();
        let mut baseline = 0.0;

        // Positive surprise
        mods.update(1.0, 0.5, &mut baseline);
        assert!(mods.reward_signal > 0.0, "Positive reward should give positive RPE");
        assert!(mods.arousal_signal > 0.0, "Reward should increase arousal");

        // After baseline adjusts, same reward gives smaller RPE
        let first_rpe = mods.reward_signal;
        mods.update(1.0, 0.5, &mut baseline);
        assert!(mods.reward_signal < first_rpe, "RPE should decrease as baseline adapts");
    }
}
