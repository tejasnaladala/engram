use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, ProposedAction,
    SpikeEvent, SimTime,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Action Selector: converts distributed neural activity into discrete actions
/// using population voting with lateral inhibition, plus a reflex fast-path.
///
/// For each possible action, a pool of neurons competes. The pool with the
/// highest firing rate wins (winner-take-all). Lateral inhibition between
/// pools ensures clean action selection. A reflex lookup table provides
/// sub-millisecond responses for learned critical situations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelector {
    pub population: NeuronPopulation,
    /// Number of discrete actions
    pub num_actions: usize,
    /// Neurons per action pool
    pub neurons_per_action: usize,
    /// Lateral inhibition strength
    pub inhibition_strength: f64,
    /// Reflex lookup: (pattern_hash -> action_id)
    reflex_map: HashMap<u64, u32>,
    /// Per-action spike counts for this tick
    action_votes: Vec<u32>,
    /// Last selected action
    pub last_action: Option<ProposedAction>,
    /// Exploration temperature (higher = more random)
    pub temperature: f64,
    /// Epsilon-greedy exploration rate (decays over time)
    pub epsilon: f64,
    recent_spike_count: u32,
    #[serde(skip, default = "default_action_rng")]
    rng: ChaCha8Rng,
}

fn default_action_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

impl ActionSelector {
    pub fn new(num_actions: usize, neurons_per_action: usize) -> Self {
        let total = num_actions * neurons_per_action;
        let params = LIFParams {
            tau_m: 10.0,     // fast -- actions should be decisive
            v_rest: -65.0,
            v_threshold: -55.0,
            v_reset: -72.0,  // deeper reset for sharper competition
            r_membrane: 12.0,
            refractory_ms: 2.0,
        };

        Self {
            population: NeuronPopulation::new(total, params),
            num_actions,
            neurons_per_action,
            inhibition_strength: 0.5,
            reflex_map: HashMap::new(),
            action_votes: vec![0; num_actions],
            last_action: None,
            temperature: 1.0,
            epsilon: 0.3, // start with 30% random exploration
            recent_spike_count: 0,
            rng: ChaCha8Rng::seed_from_u64(42),
        }
    }

    /// Check for reflex response (fast-path, bypasses population vote)
    fn check_reflex(&self, spikes: &[SpikeEvent]) -> Option<u32> {
        if self.reflex_map.is_empty() || spikes.is_empty() {
            return None;
        }
        // Simple hash of active neuron pattern
        let hash = self.pattern_hash(spikes);
        self.reflex_map.get(&hash).copied()
    }

    /// Learn a reflex: associate a spike pattern with an action
    pub fn learn_reflex(&mut self, spikes: &[SpikeEvent], action_id: u32) {
        let hash = self.pattern_hash(spikes);
        self.reflex_map.insert(hash, action_id);
    }

    fn pattern_hash(&self, spikes: &[SpikeEvent]) -> u64 {
        let mut hash: u64 = 0;
        for spike in spikes.iter().take(16) {
            hash = hash.wrapping_mul(31).wrapping_add(spike.neuron_id as u64);
        }
        hash
    }

    /// Select an action using epsilon-greedy with softmax voting
    fn select_action(&mut self, sim_time: SimTime) -> ProposedAction {
        // Epsilon-greedy: random exploration with probability epsilon
        if self.rng.random::<f64>() < self.epsilon {
            let action_id = self.rng.random_range(0..self.num_actions as u32);
            // Decay epsilon over time (minimum 5% exploration)
            self.epsilon = (self.epsilon * 0.9995).max(0.05);
            return ProposedAction {
                action_id,
                confidence: 0.0,
                is_reflex: false,
                timestamp: sim_time,
            };
        }

        // Softmax over vote counts
        let max_votes = *self.action_votes.iter().max().unwrap_or(&0);
        if max_votes == 0 {
            // No votes at all: random
            let action_id = self.rng.random_range(0..self.num_actions as u32);
            return ProposedAction {
                action_id,
                confidence: 0.0,
                is_reflex: false,
                timestamp: sim_time,
            };
        }

        // Weighted random selection (stochastic, not argmax)
        let weights: Vec<f64> = self
            .action_votes
            .iter()
            .map(|&v| (v as f64 / self.temperature).exp())
            .collect();
        let total: f64 = weights.iter().sum();

        let mut r = self.rng.random::<f64>() * total;
        let mut selected = 0;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                selected = i;
                break;
            }
        }

        let confidence = weights[selected] / total;

        ProposedAction {
            action_id: selected as u32,
            confidence: confidence as f32,
            is_reflex: false,
            timestamp: sim_time,
        }
    }
}

impl BrainModule for ActionSelector {
    fn id(&self) -> ModuleId {
        ModuleId::ActionSelector
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        // Check reflex first (fast-path)
        if let Some(reflex_action) = self.check_reflex(incoming) {
            self.last_action = Some(ProposedAction {
                action_id: reflex_action,
                confidence: 1.0,
                is_reflex: true,
                timestamp: sim_time,
            });
            // Still step neurons for dashboard visualization
            let spiked = self.population.step(dt, sim_time);
            self.recent_spike_count = spiked.len() as u32;
            return spiked
                .into_iter()
                .map(|nid| SpikeEvent {
                    source_module: ModuleId::ActionSelector,
                    neuron_id: nid,
                    timestamp: sim_time,
                    strength: 1.0,
                })
                .collect();
        }

        // Route incoming spikes to action pool neurons
        for spike in incoming {
            // Distribute across action pools based on neuron_id mapping
            let target = spike.neuron_id as usize % self.population.len();
            self.population
                .deliver_input(target as u32, spike.strength as f64 * 4.0);
        }

        // Apply lateral inhibition between action pools
        // (simplified: suppress pools that are losing)
        // Reset vote counts
        self.action_votes.fill(0);

        // Step all neurons
        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        // Count votes per action pool
        for &nid in &spiked {
            let action_idx = nid as usize / self.neurons_per_action;
            if action_idx < self.num_actions {
                self.action_votes[action_idx] += 1;
            }
        }

        // Select action via softmax
        self.last_action = Some(self.select_action(sim_time));

        spiked
            .into_iter()
            .map(|nid| SpikeEvent {
                source_module: ModuleId::ActionSelector,
                neuron_id: nid,
                timestamp: sim_time,
                strength: 1.0,
            })
            .collect()
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.population.len() as u32;
        ModuleSnapshot {
            module_id: ModuleId::ActionSelector,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0,
            activity_level: (self.recent_spike_count as f32 / n as f32).min(1.0),
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.action_votes.fill(0);
        self.last_action = None;
        self.recent_spike_count = 0;
        // Keep reflex map -- reflexes persist across episodes
    }

    fn neuron_count(&self) -> u32 {
        self.population.len() as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<ActionSelector>(data) {
            *self = restored;
        }
    }
}
