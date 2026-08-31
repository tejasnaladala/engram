use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use engram_core::{
    BrainModule, LearningRule, MemoryFormation, Neuromodulators, RuntimeSnapshot, SpikeBuffer,
    SpikeEvent, SynapseMatrix, ThreeFactorSTDP, VetoEvent,
};
use engram_modules::{
    action_selector::ActionSelector, associative_memory::AssociativeMemory,
    episodic_memory::EpisodicMemory, predictive_error::PredictiveError,
    safety_kernel::SafetyKernel, sensory_encoder::SensoryEncoder,
};

use crate::config::RuntimeConfig;
use crate::metrics::MetricsTracker;

/// The Engram cognitive runtime -- orchestrates all brain modules through
/// a 10-step cognitive loop per simulation tick.
#[derive(Clone)]
pub struct EngramRuntime {
    pub config: RuntimeConfig,

    // Brain modules
    pub sensory: SensoryEncoder,
    pub predictive: PredictiveError,
    pub associative: AssociativeMemory,
    pub episodic: EpisodicMemory,
    pub action_selector: ActionSelector,
    pub safety: SafetyKernel,

    // Inter-module synapses with three-factor learning
    pub syn_sensory_to_assoc: SynapseMatrix,
    pub learn_sensory_to_assoc: ThreeFactorSTDP,

    pub syn_sensory_to_pred: SynapseMatrix,
    pub learn_sensory_to_pred: ThreeFactorSTDP,

    pub syn_assoc_to_pred: SynapseMatrix,
    pub learn_assoc_to_pred: ThreeFactorSTDP,

    pub syn_assoc_to_action: SynapseMatrix,
    pub learn_assoc_to_action: ThreeFactorSTDP,

    // Neuromodulatory system
    pub modulators: Neuromodulators,
    pub reward_baseline: f64,

    // Spike buffer for dashboard
    pub spike_buffer: SpikeBuffer,

    // State
    pub sim_time: f64,
    pub running: bool,
    pub total_reward: f64,
    pub current_reward: f64,
    reward_pending: bool,
    learning_enabled: bool,
    pub current_observation: Vec<f64>,
    pub current_action: Option<u32>,
    pub agent_position: Option<(u32, u32)>,

    // Metrics
    pub tracker: MetricsTracker,

    // Dashboard data (accumulated between snapshots)
    pending_vetoes: Vec<VetoEvent>,
    pending_formations: Vec<MemoryFormation>,
}

impl EngramRuntime {
    pub fn new(config: RuntimeConfig) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        let sensory_count = config.input_dims * config.sensory_neurons_per_dim;
        let action_count = config.num_actions * config.neurons_per_action;

        // Create brain modules
        let sensory = SensoryEncoder::new(
            config.input_dims,
            config.sensory_neurons_per_dim,
            config.seed,
        );
        let predictive = PredictiveError::new(config.pred_error_neurons, sensory_count);
        let associative = AssociativeMemory::new(
            config.assoc_neurons,
            config.sdm_locations,
            config.sdm_data_width,
            config.seed + 1,
        );
        let episodic = EpisodicMemory::new(config.episodic_neurons, config.max_episodes);
        let action_selector = ActionSelector::new(config.num_actions, config.neurons_per_action);
        let safety = SafetyKernel::new(config.safety_neurons);

        // Create inter-module synapses
        let syn_sensory_to_assoc = SynapseMatrix::random_sparse(
            sensory_count as u32,
            config.assoc_neurons as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let learn_sensory_to_assoc = ThreeFactorSTDP::new(
            sensory_count,
            config.assoc_neurons,
            syn_sensory_to_assoc.nnz(),
        );

        let syn_sensory_to_pred = SynapseMatrix::random_sparse(
            sensory_count as u32,
            config.pred_error_neurons as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let learn_sensory_to_pred = ThreeFactorSTDP::new(
            sensory_count,
            config.pred_error_neurons,
            syn_sensory_to_pred.nnz(),
        );

        let syn_assoc_to_pred = SynapseMatrix::random_sparse(
            config.assoc_neurons as u32,
            config.pred_error_neurons as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let learn_assoc_to_pred = ThreeFactorSTDP::new(
            config.assoc_neurons,
            config.pred_error_neurons,
            syn_assoc_to_pred.nnz(),
        );

        let syn_assoc_to_action = SynapseMatrix::random_sparse(
            config.assoc_neurons as u32,
            action_count as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let learn_assoc_to_action = ThreeFactorSTDP::with_params(
            config.assoc_neurons,
            action_count,
            syn_assoc_to_action.nnz(),
            500.0, // shorter eligibility for action pathway
            0.008, // higher learning rate for action selection
        );

        Self {
            config,
            sensory,
            predictive,
            associative,
            episodic,
            action_selector,
            safety,
            syn_sensory_to_assoc,
            learn_sensory_to_assoc,
            syn_sensory_to_pred,
            learn_sensory_to_pred,
            syn_assoc_to_pred,
            learn_assoc_to_pred,
            syn_assoc_to_action,
            learn_assoc_to_action,
            modulators: Neuromodulators::default(),
            reward_baseline: 0.0,
            spike_buffer: SpikeBuffer::new(5000),
            sim_time: 0.0,
            running: true,
            total_reward: 0.0,
            current_reward: 0.0,
            reward_pending: false,
            learning_enabled: true,
            current_observation: Vec::new(),
            current_action: None,
            agent_position: None,
            tracker: MetricsTracker::new(),
            pending_vetoes: Vec::new(),
            pending_formations: Vec::new(),
        }
    }

    /// Set the current observation for the next tick
    pub fn set_observation(&mut self, obs: &[f64]) {
        self.current_observation = obs.to_vec();
        self.sensory.set_observation(obs);
    }

    /// Queue incremental environment feedback for one learning update.
    pub fn set_reward(&mut self, reward: f64) {
        if self.reward_pending {
            self.current_reward += reward;
        } else {
            self.current_reward = reward;
        }
        self.total_reward += reward;
        self.reward_pending = true;

        if self.learning_enabled && self.config.replay_enabled {
            self.episodic.attach_reward_to_latest_frame(reward);
        }
        if self.learning_enabled && self.config.safety_enabled && reward < -0.5 {
            if let Some(action) = self.current_action {
                self.safety.learn_from_negative(action, reward);
            }
        }
    }

    /// Whether environment feedback is waiting to be consumed.
    pub fn has_pending_reward(&self) -> bool {
        self.reward_pending
    }

    /// Enable or disable every persistent learning path.
    pub fn set_learning_enabled(&mut self, enabled: bool) {
        self.learning_enabled = enabled;
        self.associative.set_learning_enabled(enabled);
        self.episodic.set_learning_enabled(enabled);
        self.action_selector.set_learning_enabled(enabled);
    }

    /// Whether persistent learning is enabled.
    pub fn learning_enabled(&self) -> bool {
        self.learning_enabled
    }

    /// Deterministic non-cryptographic fingerprint of persistent learning state.
    pub fn learning_state_hash(&self) -> u64 {
        let core_state = engram_core::checkpoint::serialize(&(
            &self.syn_sensory_to_assoc,
            &self.learn_sensory_to_assoc,
            &self.syn_sensory_to_pred,
            &self.learn_sensory_to_pred,
            &self.syn_assoc_to_pred,
            &self.learn_assoc_to_pred,
            &self.syn_assoc_to_action,
            &self.learn_assoc_to_action,
            self.reward_baseline.to_bits(),
        ))
        .expect("serializing core learning state should succeed");
        let chunks = [
            core_state,
            self.associative.learning_state_bytes(),
            self.episodic.learning_state_bytes(),
            self.action_selector.learning_state_bytes(),
            self.safety.learning_state_bytes(),
        ];

        let mut hash = 0xcbf29ce484222325_u64;
        for chunk in chunks {
            for byte in (chunk.len() as u64)
                .to_le_bytes()
                .iter()
                .chain(chunk.iter())
            {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    /// Set agent position for dashboard
    pub fn set_agent_position(&mut self, x: u32, y: u32) {
        self.agent_position = Some((x, y));
    }

    /// Execute one tick of the 10-step cognitive loop.
    /// Returns the selected action ID.
    pub fn step(&mut self) -> u32 {
        let dt = self.config.dt;
        let sim_time = self.sim_time;

        // Feedback belongs to the action completed before this tick. Apply it
        // to the existing eligibility traces before new observation/action
        // spikes can enter those traces.
        self.apply_pending_reward();

        // === STEP 1: Sensory Encoding ===
        let sensory_spikes = self.sensory.step(dt, sim_time, &[]);
        self.spike_buffer.extend(sensory_spikes.iter().cloned());

        // === STEP 2: Route to Associative Memory ===
        let sensory_ids: Vec<u32> = sensory_spikes.iter().map(|s| s.neuron_id).collect();
        let assoc_inputs = self.syn_sensory_to_assoc.propagate(&sensory_ids);
        // Deliver currents to associative memory neurons
        for (post_id, current) in &assoc_inputs {
            self.associative
                .population
                .deliver_input(*post_id, *current);
        }

        // === STEP 3: Route to Predictive Error (actual) ===
        let pred_inputs = self.syn_sensory_to_pred.propagate(&sensory_ids);
        for (post_id, current) in &pred_inputs {
            self.predictive.population.deliver_input(*post_id, *current);
        }

        // === STEP 4: Associative Memory Step ===
        let assoc_spikes = self.associative.step(dt, sim_time, &sensory_spikes);
        self.spike_buffer.extend(assoc_spikes.iter().cloned());

        // Route associative predictions to predictive error module
        let assoc_ids: Vec<u32> = assoc_spikes.iter().map(|s| s.neuron_id).collect();
        let pred_from_assoc = self.syn_assoc_to_pred.propagate(&assoc_ids);
        for (post_id, current) in &pred_from_assoc {
            self.predictive.population.deliver_input(*post_id, *current);
        }

        // === STEP 5: Predictive Error Step ===
        let mut pred_input_spikes = sensory_spikes.clone();
        pred_input_spikes.extend(assoc_spikes.iter().cloned());
        let pred_spikes = self.predictive.step(dt, sim_time, &pred_input_spikes);
        self.spike_buffer.extend(pred_spikes.iter().cloned());

        // === STEP 6: Update the current surprise signal ===
        // Reward modulation was consumed at the tick boundary above.
        self.modulators.reward_signal = 0.0;
        self.modulators.surprise_signal = self.predictive.error;

        // === STEP 7: Action Selection ===
        // Route associative spikes to action selector
        let action_inputs = self.syn_assoc_to_action.propagate(&assoc_ids);
        for (post_id, current) in &action_inputs {
            self.action_selector
                .population
                .deliver_input(*post_id, *current);
        }
        let action_spikes = self.action_selector.step(dt, sim_time, &assoc_spikes);
        self.spike_buffer.extend(action_spikes.iter().cloned());

        let proposed =
            self.action_selector
                .last_action
                .clone()
                .unwrap_or(engram_core::ProposedAction {
                    action_id: 0,
                    confidence: 0.0,
                    is_reflex: false,
                    timestamp: sim_time,
                });

        // === STEP 8: Safety Evaluation ===
        let mut final_action = proposed.action_id;
        if self.config.safety_enabled {
            self.safety.set_state(&self.current_observation);
            let safety_spikes = self.safety.step(dt, sim_time, &pred_spikes);
            self.spike_buffer.extend(safety_spikes.iter().cloned());

            if let Some(veto) = self.safety.evaluate(&proposed, sim_time) {
                self.tracker.record_veto();
                self.pending_vetoes.push(veto);
                final_action = 0; // default safe action
            }
        }

        self.current_action = Some(final_action);

        // === STEP 9: Episodic Recording & Replay ===
        if self.config.replay_enabled {
            if self.learning_enabled {
                self.episodic
                    .record_frame(&sensory_spikes, 0.0, self.predictive.error);
            }
            let episodic_spikes = self.episodic.step(dt, sim_time, &assoc_spikes);
            self.spike_buffer.extend(episodic_spikes.iter().cloned());
        }

        // === STEP 10: Three-Factor Learning Rule Updates ===
        // Each pathway learns via eligibility traces * neuromodulatory signal
        let assoc_spike_ids: Vec<u32> = assoc_spikes.iter().map(|s| s.neuron_id).collect();
        let pred_spike_ids: Vec<u32> = pred_spikes.iter().map(|s| s.neuron_id).collect();
        let action_spike_ids: Vec<u32> = action_spikes.iter().map(|s| s.neuron_id).collect();

        if self.learning_enabled {
            self.apply_learning_updates(
                dt,
                &sensory_ids,
                &assoc_spike_ids,
                &pred_spike_ids,
                &action_spike_ids,
            );
        }

        // === Update Metrics ===
        let total_spikes =
            sensory_spikes.len() + assoc_spikes.len() + pred_spikes.len() + action_spikes.len();
        self.tracker.record_spikes(total_spikes as u64);
        self.tracker.record_tick();
        self.tracker.add_energy(total_spikes as f64 * 0.001);

        let total_synapses = self.syn_sensory_to_assoc.nnz()
            + self.syn_sensory_to_pred.nnz()
            + self.syn_assoc_to_pred.nnz()
            + self.syn_assoc_to_action.nnz();
        self.tracker.set_active_synapses(total_synapses as u64);

        // Advance simulation time
        self.sim_time += dt;
        self.tracker.set_sim_time(self.sim_time);

        // Collect memory formations from modules
        self.pending_formations
            .extend(self.associative.take_formations());
        self.pending_formations
            .extend(self.episodic.take_formations());

        final_action
    }

    fn apply_learning_updates(
        &mut self,
        dt: f64,
        sensory_ids: &[u32],
        assoc_spike_ids: &[u32],
        pred_spike_ids: &[u32],
        action_spike_ids: &[u32],
    ) {
        self.learn_sensory_to_assoc.apply(
            dt,
            &mut self.syn_sensory_to_assoc,
            sensory_ids,
            assoc_spike_ids,
            &self.modulators,
        );
        self.learn_sensory_to_pred.apply(
            dt,
            &mut self.syn_sensory_to_pred,
            sensory_ids,
            pred_spike_ids,
            &self.modulators,
        );
        self.learn_assoc_to_pred.apply(
            dt,
            &mut self.syn_assoc_to_pred,
            assoc_spike_ids,
            pred_spike_ids,
            &self.modulators,
        );
        self.learn_assoc_to_action.apply(
            dt,
            &mut self.syn_assoc_to_action,
            assoc_spike_ids,
            action_spike_ids,
            &self.modulators,
        );
    }

    fn apply_pending_reward(&mut self) {
        if !self.reward_pending {
            return;
        }

        if self.learning_enabled {
            self.modulators.update(
                self.current_reward,
                self.predictive.error,
                &mut self.reward_baseline,
            );
            self.apply_learning_updates(0.0, &[], &[], &[], &[]);
        }

        self.clear_pending_reward();
    }

    fn clear_pending_reward(&mut self) {
        self.current_reward = 0.0;
        self.reward_pending = false;
        self.modulators.reward_signal = 0.0;
    }

    /// Generate a snapshot for the dashboard
    pub fn snapshot(&mut self) -> RuntimeSnapshot {
        let modules = vec![
            self.sensory.snapshot(),
            self.associative.snapshot(),
            self.predictive.snapshot(),
            self.episodic.snapshot(),
            self.action_selector.snapshot(),
            self.safety.snapshot(),
        ];

        let recent_spikes: Vec<SpikeEvent> = self
            .spike_buffer
            .recent(50.0, self.sim_time)
            .into_iter()
            .cloned()
            .collect();

        let vetoes = std::mem::take(&mut self.pending_vetoes);
        let formations = std::mem::take(&mut self.pending_formations);

        RuntimeSnapshot {
            metrics: self.tracker.metrics.clone(),
            modules,
            recent_spikes,
            recent_vetoes: vetoes,
            prediction_error: self.predictive.error,
            memory_formations: formations,
            current_action: self.current_action,
            total_reward: self.total_reward,
            agent_position: self.agent_position,
        }
    }

    /// Reset all modules for a new episode
    pub fn reset_episode(&mut self) {
        self.apply_pending_reward();
        self.sensory.reset();
        self.predictive.reset();
        // Don't reset associative memory -- it persists across episodes
        self.episodic.reset();
        self.action_selector.reset();
        self.safety.reset();
        self.current_action = None;
        self.spike_buffer.clear();
    }

    /// Full reset including memories
    pub fn full_reset(&mut self) {
        let config = self.config.clone();
        *self = Self::new(config);
    }

    /// Get current prediction error
    pub fn prediction_error(&self) -> f64 {
        self.predictive.error
    }

    /// Get current tick count
    pub fn tick(&self) -> u64 {
        self.tracker.metrics.tick
    }

    /// Total spike count
    pub fn total_spikes(&self) -> u64 {
        self.tracker.metrics.total_spikes
    }

    /// Total veto count
    pub fn total_vetoes(&self) -> u64 {
        self.tracker.metrics.total_vetoes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> RuntimeConfig {
        RuntimeConfig {
            input_dims: 2,
            sensory_neurons_per_dim: 2,
            num_actions: 2,
            neurons_per_action: 2,
            assoc_neurons: 8,
            sdm_locations: 16,
            sdm_data_width: 8,
            pred_error_neurons: 4,
            episodic_neurons: 4,
            safety_neurons: 4,
            max_episodes: 4,
            synapse_density: 0.5,
            replay_enabled: false,
            ..RuntimeConfig::default()
        }
    }

    #[test]
    fn reward_is_counted_and_consumed_exactly_once() {
        let mut runtime = EngramRuntime::new(tiny_config());
        runtime.set_observation(&[0.25, 0.75]);

        runtime.set_reward(1.25);
        assert_eq!(runtime.total_reward, 1.25);
        assert!(runtime.has_pending_reward());

        runtime.step();
        assert!(!runtime.has_pending_reward());
        let baseline_after_reward = runtime.reward_baseline;

        runtime.step();
        assert_eq!(runtime.total_reward, 1.25);
        assert_eq!(runtime.reward_baseline, baseline_after_reward);
    }

    #[test]
    fn episode_end_flushes_terminal_reward_once() {
        let mut runtime = EngramRuntime::new(tiny_config());
        runtime.set_observation(&[0.25, 0.75]);
        runtime.step();

        runtime.set_reward(2.0);
        runtime.reset_episode();

        assert!(!runtime.has_pending_reward());
        assert_eq!(runtime.total_reward, 2.0);
        assert!(runtime.reward_baseline > 0.0);
    }

    #[test]
    fn negative_reward_is_attributed_to_the_completed_action() {
        let mut runtime = EngramRuntime::new(tiny_config());
        runtime.set_observation(&[0.25, 0.75]);
        runtime.step();
        let before = runtime.learning_state_hash();

        runtime.set_reward(-1.0);

        assert_ne!(runtime.learning_state_hash(), before);
    }

    #[test]
    fn frozen_evaluation_preserves_learning_state() {
        let mut config = tiny_config();
        config.replay_enabled = true;
        let mut runtime = EngramRuntime::new(config);
        runtime.set_observation(&[0.2, 0.8]);
        runtime.step();
        runtime.set_learning_enabled(false);
        let before = runtime.learning_state_hash();

        for reward in [1.0, -0.5, 0.25] {
            runtime.set_observation(&[0.2, 0.8]);
            runtime.set_reward(reward);
            runtime.step();
        }
        runtime.reset_episode();

        assert_eq!(runtime.learning_state_hash(), before);
    }

    #[test]
    fn pending_reward_is_applied_before_next_tick_spikes() {
        let mut runtime = EngramRuntime::new(tiny_config());
        runtime.set_observation(&[0.95, 0.85]);
        for _ in 0..20 {
            runtime.step();
        }

        let mut explicitly_flushed = runtime.clone();
        explicitly_flushed.set_reward(1.0);
        explicitly_flushed.apply_pending_reward();

        let mut advanced = runtime.clone();
        advanced.set_reward(1.0);
        advanced.set_observation(&[0.05, 0.15]);
        advanced.step();

        assert_eq!(
            advanced.syn_sensory_to_assoc.values,
            explicitly_flushed.syn_sensory_to_assoc.values,
        );
        assert_eq!(
            advanced.syn_sensory_to_pred.values,
            explicitly_flushed.syn_sensory_to_pred.values,
        );
        assert_eq!(
            advanced.syn_assoc_to_pred.values,
            explicitly_flushed.syn_assoc_to_pred.values,
        );
        assert_eq!(
            advanced.syn_assoc_to_action.values,
            explicitly_flushed.syn_assoc_to_action.values,
        );
    }

    #[test]
    fn full_reset_restores_the_seeded_learning_state() {
        let config = tiny_config();
        let initial = EngramRuntime::new(config.clone());
        let initial_hash = initial.learning_state_hash();
        let mut runtime = EngramRuntime::new(config);

        runtime.set_observation(&[0.95, 0.85]);
        runtime.step();
        runtime.set_reward(-1.0);
        assert_ne!(runtime.learning_state_hash(), initial_hash);

        runtime.set_learning_enabled(false);
        runtime.full_reset();

        assert_eq!(runtime.learning_state_hash(), initial_hash);
        assert!(runtime.learning_enabled());
        assert_eq!(runtime.total_reward, 0.0);
        assert_eq!(runtime.sim_time, 0.0);
        assert!(!runtime.has_pending_reward());
    }
}
