use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use engram_core::{
    BrainModule, MemoryFormation, ModuleId, ModuleSnapshot, RuntimeSnapshot, STDPParams,
    STDPState, SpikeBuffer, SpikeEvent, SynapseMatrix, VetoEvent,
};
use engram_modules::{
    action_selector::ActionSelector,
    associative_memory::AssociativeMemory,
    episodic_memory::EpisodicMemory,
    predictive_error::PredictiveError,
    safety_kernel::SafetyKernel,
    sensory_encoder::SensoryEncoder,
};

use crate::config::RuntimeConfig;
use crate::metrics::MetricsTracker;

/// The Engram cognitive runtime — orchestrates all brain modules through
/// a 10-step cognitive loop per simulation tick.
pub struct EngramRuntime {
    pub config: RuntimeConfig,

    // Brain modules
    pub sensory: SensoryEncoder,
    pub predictive: PredictiveError,
    pub associative: AssociativeMemory,
    pub episodic: EpisodicMemory,
    pub action_selector: ActionSelector,
    pub safety: SafetyKernel,

    // Inter-module synapses with STDP
    pub syn_sensory_to_assoc: SynapseMatrix,
    pub stdp_sensory_to_assoc: STDPState,

    pub syn_sensory_to_pred: SynapseMatrix,
    pub stdp_sensory_to_pred: STDPState,

    pub syn_assoc_to_pred: SynapseMatrix,
    pub stdp_assoc_to_pred: STDPState,

    pub syn_assoc_to_action: SynapseMatrix,
    pub stdp_assoc_to_action: STDPState,

    // Spike buffer for dashboard
    pub spike_buffer: SpikeBuffer,

    // State
    pub sim_time: f64,
    pub running: bool,
    pub total_reward: f64,
    pub current_reward: f64,
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
        let stdp_sensory_to_assoc = STDPState::new(
            sensory_count,
            config.assoc_neurons,
            STDPParams::default(),
        );

        let syn_sensory_to_pred = SynapseMatrix::random_sparse(
            sensory_count as u32,
            config.pred_error_neurons as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let stdp_sensory_to_pred = STDPState::new(
            sensory_count,
            config.pred_error_neurons,
            STDPParams::default(),
        );

        let syn_assoc_to_pred = SynapseMatrix::random_sparse(
            config.assoc_neurons as u32,
            config.pred_error_neurons as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let stdp_assoc_to_pred = STDPState::new(
            config.assoc_neurons,
            config.pred_error_neurons,
            STDPParams::default(),
        );

        let syn_assoc_to_action = SynapseMatrix::random_sparse(
            config.assoc_neurons as u32,
            action_count as u32,
            config.synapse_density,
            config.w_init_max,
            &mut rng,
        );
        let stdp_assoc_to_action = STDPState::new(
            config.assoc_neurons,
            action_count,
            STDPParams::default(),
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
            stdp_sensory_to_assoc,
            syn_sensory_to_pred,
            stdp_sensory_to_pred,
            syn_assoc_to_pred,
            stdp_assoc_to_pred,
            syn_assoc_to_action,
            stdp_assoc_to_action,
            spike_buffer: SpikeBuffer::new(5000),
            sim_time: 0.0,
            running: true,
            total_reward: 0.0,
            current_reward: 0.0,
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

    /// Set the reward signal
    pub fn set_reward(&mut self, reward: f64) {
        self.current_reward = reward;
        self.total_reward += reward;
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
            self.predictive
                .population
                .deliver_input(*post_id, *current);
        }

        // === STEP 4: Associative Memory Step ===
        let assoc_spikes = self.associative.step(dt, sim_time, &sensory_spikes);
        self.spike_buffer.extend(assoc_spikes.iter().cloned());

        // Route associative predictions to predictive error module
        let assoc_ids: Vec<u32> = assoc_spikes.iter().map(|s| s.neuron_id).collect();
        let pred_from_assoc = self.syn_assoc_to_pred.propagate(&assoc_ids);
        for (post_id, current) in &pred_from_assoc {
            self.predictive
                .population
                .deliver_input(*post_id, *current);
        }

        // === STEP 5: Predictive Error Step ===
        let mut pred_input_spikes = sensory_spikes.clone();
        pred_input_spikes.extend(assoc_spikes.iter().cloned());
        let pred_spikes = self.predictive.step(dt, sim_time, &pred_input_spikes);
        self.spike_buffer.extend(pred_spikes.iter().cloned());

        // === STEP 6: Error-Modulated STDP ===
        let lr_mod = self.predictive.learning_rate_modifier();
        self.stdp_sensory_to_assoc.set_learning_rate_mod(lr_mod);
        self.stdp_sensory_to_pred.set_learning_rate_mod(lr_mod);
        self.stdp_assoc_to_pred.set_learning_rate_mod(lr_mod);
        self.stdp_assoc_to_action.set_learning_rate_mod(lr_mod);

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

        let proposed = self
            .action_selector
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

            // Learn from negative reward
            if self.current_reward < -0.5 {
                self.safety
                    .learn_from_negative(proposed.action_id, self.current_reward);
            }
        }

        self.current_action = Some(final_action);

        // === STEP 9: Episodic Recording & Replay ===
        if self.config.replay_enabled {
            self.episodic.record_frame(
                &sensory_spikes,
                self.current_reward,
                self.predictive.error,
            );
            let episodic_spikes = self.episodic.step(dt, sim_time, &assoc_spikes);
            self.spike_buffer.extend(episodic_spikes.iter().cloned());
        }

        // === STEP 10: STDP Weight Updates ===
        let assoc_spike_ids: Vec<u32> = assoc_spikes.iter().map(|s| s.neuron_id).collect();
        let pred_spike_ids: Vec<u32> = pred_spikes.iter().map(|s| s.neuron_id).collect();
        let action_spike_ids: Vec<u32> = action_spikes.iter().map(|s| s.neuron_id).collect();

        self.stdp_sensory_to_assoc.apply(
            dt,
            &mut self.syn_sensory_to_assoc,
            &sensory_ids,
            &assoc_spike_ids,
        );
        self.stdp_sensory_to_pred.apply(
            dt,
            &mut self.syn_sensory_to_pred,
            &sensory_ids,
            &pred_spike_ids,
        );
        self.stdp_assoc_to_pred.apply(
            dt,
            &mut self.syn_assoc_to_pred,
            &assoc_spike_ids,
            &pred_spike_ids,
        );
        self.stdp_assoc_to_action.apply(
            dt,
            &mut self.syn_assoc_to_action,
            &assoc_spike_ids,
            &action_spike_ids,
        );

        // === Update Metrics ===
        let total_spikes = sensory_spikes.len()
            + assoc_spikes.len()
            + pred_spikes.len()
            + action_spikes.len();
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

        let recent_spikes: Vec<SpikeEvent> =
            self.spike_buffer.recent(50.0, self.sim_time).into_iter().cloned().collect();

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
        self.sensory.reset();
        self.predictive.reset();
        // Don't reset associative memory — it persists across episodes
        self.episodic.reset();
        self.action_selector.reset();
        self.safety.reset();
        self.current_reward = 0.0;
        self.current_action = None;
        self.spike_buffer.clear();
    }

    /// Full reset including memories
    pub fn full_reset(&mut self) {
        self.reset_episode();
        self.associative.reset();
        self.sim_time = 0.0;
        self.total_reward = 0.0;
        self.tracker.reset();
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
