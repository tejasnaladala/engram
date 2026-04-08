use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, ProposedAction,
    SpikeEvent, SimTime, VetoEvent, VetoReason,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A hard constraint that can never be overridden
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardConstraint {
    pub name: String,
    /// Predicate function index (we use an enum since closures aren't serializable)
    pub constraint_type: ConstraintType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Prevent moving into a cell of a specific type
    ForbidCellType(u8),
    /// Rate limit: max actions per second
    RateLimit(f64),
    /// Energy budget: max cumulative energy units
    EnergyBudget(f64),
}

/// Safety Kernel: parallel safety evaluator that can veto proposed actions.
///
/// Two-tier design:
/// 1. Hard constraints: immutable rules defined at initialization
/// 2. Learned inhibition: patterns learned from negative outcomes via STDP
///
/// The safety kernel runs conceptually in parallel with the action selector
/// and can block actions before they reach the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyKernel {
    pub population: NeuronPopulation,
    /// Hard constraint rules
    hard_constraints: Vec<HardConstraint>,
    /// Learned danger patterns: (state_hash -> (action_id, confidence))
    learned_inhibitions: HashMap<u64, Vec<(u32, f32)>>,
    /// Inhibition confidence threshold for veto
    inhibition_threshold: f32,
    /// Action timestamps for rate limiting
    action_times: Vec<SimTime>,
    /// Cumulative energy consumed
    energy_consumed: f64,
    /// Recent veto events for dashboard
    recent_vetoes: Vec<VetoEvent>,
    /// Current environment state for constraint evaluation
    current_state: Vec<f64>,
    /// Grid state for cell-type constraints
    grid_state: Vec<u8>,
    grid_width: u32,
    recent_spike_count: u32,
}

impl SafetyKernel {
    pub fn new(num_neurons: usize) -> Self {
        let params = LIFParams {
            tau_m: 5.0,       // very fast — safety must be responsive
            v_rest: -65.0,
            v_threshold: -60.0, // low threshold — sensitive
            v_reset: -70.0,
            r_membrane: 20.0,
            refractory_ms: 0.5,  // very short refractory
        };

        Self {
            population: NeuronPopulation::new(num_neurons, params),
            hard_constraints: Vec::new(),
            learned_inhibitions: HashMap::new(),
            inhibition_threshold: 0.7,
            action_times: Vec::new(),
            energy_consumed: 0.0,
            recent_vetoes: Vec::new(),
            current_state: Vec::new(),
            grid_state: Vec::new(),
            grid_width: 0,
            recent_spike_count: 0,
        }
    }

    /// Add a hard constraint
    pub fn add_constraint(&mut self, constraint: HardConstraint) {
        self.hard_constraints.push(constraint);
    }

    /// Set current environment state for constraint evaluation
    pub fn set_state(&mut self, state: &[f64]) {
        self.current_state = state.to_vec();
    }

    /// Set grid state for cell-type constraints
    pub fn set_grid(&mut self, grid: &[u8], width: u32) {
        self.grid_state = grid.to_vec();
        self.grid_width = width;
    }

    /// Evaluate a proposed action against all safety constraints.
    /// Returns None if safe, Some(VetoEvent) if vetoed.
    pub fn evaluate(&mut self, action: &ProposedAction, sim_time: SimTime) -> Option<VetoEvent> {
        // Check hard constraints
        for constraint in &self.hard_constraints {
            match &constraint.constraint_type {
                ConstraintType::ForbidCellType(cell_type) => {
                    // Check if the action would move into a forbidden cell
                    if let Some(target_cell) =
                        self.get_target_cell(action.action_id)
                    {
                        if target_cell == *cell_type {
                            return Some(VetoEvent {
                                timestamp: sim_time,
                                vetoed_action: action.clone(),
                                reason: VetoReason::HardConstraint(constraint.name.clone()),
                            });
                        }
                    }
                }
                ConstraintType::RateLimit(max_per_sec) => {
                    let recent = self
                        .action_times
                        .iter()
                        .filter(|&&t| sim_time - t < 1000.0)
                        .count();
                    if recent as f64 >= *max_per_sec {
                        return Some(VetoEvent {
                            timestamp: sim_time,
                            vetoed_action: action.clone(),
                            reason: VetoReason::HardConstraint(constraint.name.clone()),
                        });
                    }
                }
                ConstraintType::EnergyBudget(max_energy) => {
                    if self.energy_consumed >= *max_energy {
                        return Some(VetoEvent {
                            timestamp: sim_time,
                            vetoed_action: action.clone(),
                            reason: VetoReason::EnergyBudgetExceeded,
                        });
                    }
                }
            }
        }

        // Check learned inhibitions
        let state_hash = self.hash_state();
        if let Some(inhibitions) = self.learned_inhibitions.get(&state_hash) {
            for &(inhibited_action, confidence) in inhibitions {
                if inhibited_action == action.action_id && confidence >= self.inhibition_threshold {
                    return Some(VetoEvent {
                        timestamp: sim_time,
                        vetoed_action: action.clone(),
                        reason: VetoReason::LearnedInhibition { confidence },
                    });
                }
            }
        }

        // Record action time for rate limiting
        self.action_times.push(sim_time);
        // Evict old timestamps
        self.action_times.retain(|&t| sim_time - t < 2000.0);
        // Track energy
        self.energy_consumed += 1.0;

        None
    }

    /// Learn from negative reward: strengthen inhibition for this state-action pair
    pub fn learn_from_negative(&mut self, action_id: u32, negative_reward: f64) {
        let state_hash = self.hash_state();
        let confidence_delta = (-negative_reward * 0.1).min(0.3) as f32;

        let entry = self
            .learned_inhibitions
            .entry(state_hash)
            .or_insert_with(Vec::new);

        if let Some(existing) = entry.iter_mut().find(|(a, _)| *a == action_id) {
            existing.1 = (existing.1 + confidence_delta).min(1.0);
        } else {
            entry.push((action_id, confidence_delta));
        }
    }

    /// Take and clear recent vetoes (for dashboard)
    pub fn take_vetoes(&mut self) -> Vec<VetoEvent> {
        std::mem::take(&mut self.recent_vetoes)
    }

    fn hash_state(&self) -> u64 {
        let mut hash: u64 = 0;
        for (i, &val) in self.current_state.iter().enumerate().take(8) {
            let quantized = (val * 100.0) as u64;
            hash = hash.wrapping_mul(31).wrapping_add(quantized + i as u64);
        }
        hash
    }

    fn get_target_cell(&self, action_id: u32) -> Option<u8> {
        if self.grid_state.is_empty() || self.current_state.len() < 2 {
            return None;
        }
        let w = self.grid_width as i32;
        let x = (self.current_state[0] * w as f64) as i32;
        let y = (self.current_state[1] * w as f64) as i32;
        let (dx, dy) = match action_id {
            0 => (0, -1), // up
            1 => (1, 0),  // right
            2 => (0, 1),  // down
            3 => (-1, 0), // left
            _ => return None,
        };
        let nx = x + dx;
        let ny = y + dy;
        if nx < 0 || ny < 0 || nx >= w || ny >= w as i32 {
            return None;
        }
        let idx = (ny * w + nx) as usize;
        self.grid_state.get(idx).copied()
    }
}

impl BrainModule for SafetyKernel {
    fn id(&self) -> ModuleId {
        ModuleId::SafetyKernel
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        // Drive safety neurons from incoming danger signals
        for spike in incoming {
            let idx = spike.neuron_id as usize % self.population.len();
            self.population.deliver_input(idx as u32, spike.strength as f64 * 3.0);
        }

        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        spiked
            .into_iter()
            .map(|nid| SpikeEvent {
                source_module: ModuleId::SafetyKernel,
                neuron_id: nid,
                timestamp: sim_time,
                strength: 1.0,
            })
            .collect()
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.population.len() as u32;
        ModuleSnapshot {
            module_id: ModuleId::SafetyKernel,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0,
            activity_level: if !self.recent_vetoes.is_empty() {
                1.0
            } else {
                (self.recent_spike_count as f32 / n as f32).min(1.0)
            },
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.action_times.clear();
        self.recent_vetoes.clear();
        self.recent_spike_count = 0;
        // Keep learned inhibitions — they persist across episodes
    }

    fn neuron_count(&self) -> u32 {
        self.population.len() as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<SafetyKernel>(data) {
            *self = restored;
        }
    }
}
