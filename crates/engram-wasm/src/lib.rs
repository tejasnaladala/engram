use wasm_bindgen::prelude::*;
use engram_core::{ModuleId, RuntimeSnapshot};
use engram_modules::{
    sensory_encoder::SensoryEncoder,
    predictive_error::PredictiveError,
    associative_memory::AssociativeMemory,
    episodic_memory::EpisodicMemory,
    action_selector::ActionSelector,
    safety_kernel::SafetyKernel,
};
use engram_core::{BrainModule, SynapseMatrix, STDPState, STDPParams, SpikeBuffer};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// WASM-compatible runtime (synchronous, single-threaded)
#[wasm_bindgen]
pub struct WasmRuntime {
    sensory: SensoryEncoder,
    predictive: PredictiveError,
    associative: AssociativeMemory,
    episodic: EpisodicMemory,
    action_selector: ActionSelector,
    safety: SafetyKernel,

    syn_s2a: SynapseMatrix,
    stdp_s2a: STDPState,

    spike_buffer: SpikeBuffer,
    sim_time: f64,
    dt: f64,
    tick_count: u64,
    total_reward: f64,
    current_action: u32,
    prediction_error: f64,

    // Grid world state
    grid: Vec<u8>,
    grid_size: u32,
    agent_x: u32,
    agent_y: u32,
    target_x: u32,
    target_y: u32,
}

#[wasm_bindgen]
impl WasmRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new(grid_size: u32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let input_dims = 8;
        let sensory_count = input_dims * 16;

        let sensory = SensoryEncoder::new(input_dims, 16, 42);
        let predictive = PredictiveError::new(64, sensory_count);
        let associative = AssociativeMemory::new(256, 500, 128, 43);
        let episodic = EpisodicMemory::new(64, 50);
        let action_selector = ActionSelector::new(4, 32);
        let safety = SafetyKernel::new(32);

        let syn_s2a = SynapseMatrix::random_sparse(
            sensory_count as u32, 256, 0.3, 0.5, &mut rng,
        );
        let stdp_s2a = STDPState::new(sensory_count, 256, STDPParams::default());

        let grid = vec![0u8; (grid_size * grid_size) as usize];

        Self {
            sensory, predictive, associative, episodic, action_selector, safety,
            syn_s2a, stdp_s2a,
            spike_buffer: SpikeBuffer::new(2000),
            sim_time: 0.0,
            dt: 1.0,
            tick_count: 0,
            total_reward: 0.0,
            current_action: 0,
            prediction_error: 0.0,
            grid,
            grid_size,
            agent_x: 1,
            agent_y: 1,
            target_x: grid_size - 2,
            target_y: grid_size - 2,
        }
    }

    /// Set a cell in the grid (0=empty, 1=wall, 2=reward, 3=hazard)
    pub fn set_cell(&mut self, x: u32, y: u32, cell_type: u8) {
        let idx = (y * self.grid_size + x) as usize;
        if idx < self.grid.len() {
            self.grid[idx] = cell_type;
        }
    }

    /// Run one simulation tick, returns action taken
    pub fn step(&mut self) -> u32 {
        // Build observation from grid neighborhood
        let obs = self.build_observation();
        self.sensory.set_observation(&obs);

        // Run sensory encoding
        let sensory_spikes = self.sensory.step(self.dt, self.sim_time, &[]);

        // Route to associative memory
        let sensory_ids: Vec<u32> = sensory_spikes.iter().map(|s| s.neuron_id).collect();
        let assoc_inputs = self.syn_s2a.propagate(&sensory_ids);
        for (post_id, current) in &assoc_inputs {
            self.associative.population.deliver_input(*post_id, *current);
        }

        // Associative memory step
        let assoc_spikes = self.associative.step(self.dt, self.sim_time, &sensory_spikes);

        // Predictive error
        let mut all_spikes = sensory_spikes.clone();
        all_spikes.extend(assoc_spikes.iter().cloned());
        let _pred_spikes = self.predictive.step(self.dt, self.sim_time, &all_spikes);
        self.prediction_error = self.predictive.error;

        // Action selection
        let action_spikes = self.action_selector.step(self.dt, self.sim_time, &assoc_spikes);
        let action = self.action_selector.last_action
            .as_ref()
            .map(|a| a.action_id)
            .unwrap_or(0);

        // Apply action in grid world
        let reward = self.apply_action(action);
        self.total_reward += reward;

        // STDP update
        let assoc_ids: Vec<u32> = assoc_spikes.iter().map(|s| s.neuron_id).collect();
        self.stdp_s2a.apply(self.dt, &mut self.syn_s2a, &sensory_ids, &assoc_ids);

        // Buffer spikes for visualization
        self.spike_buffer.extend(sensory_spikes);
        self.spike_buffer.extend(assoc_spikes);
        self.spike_buffer.extend(action_spikes);

        self.sim_time += self.dt;
        self.tick_count += 1;
        self.current_action = action;
        action
    }

    fn build_observation(&self) -> Vec<f64> {
        let gs = self.grid_size as f64;
        let mut obs = vec![
            self.agent_x as f64 / gs,
            self.agent_y as f64 / gs,
            self.target_x as f64 / gs,
            self.target_y as f64 / gs,
        ];
        // Add neighborhood (4 cardinal directions)
        let dirs = [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)];
        for (dx, dy) in dirs {
            let nx = self.agent_x as i32 + dx;
            let ny = self.agent_y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < self.grid_size as i32 && ny < self.grid_size as i32 {
                let cell = self.grid[(ny * self.grid_size as i32 + nx) as usize];
                obs.push(cell as f64 / 3.0);
            } else {
                obs.push(1.0); // wall
            }
        }
        obs
    }

    fn apply_action(&mut self, action: u32) -> f64 {
        let (dx, dy): (i32, i32) = match action {
            0 => (0, -1),  // up
            1 => (1, 0),   // right
            2 => (0, 1),   // down
            3 => (-1, 0),  // left
            _ => (0, 0),
        };
        let nx = self.agent_x as i32 + dx;
        let ny = self.agent_y as i32 + dy;

        if nx < 0 || ny < 0 || nx >= self.grid_size as i32 || ny >= self.grid_size as i32 {
            return -0.1; // hit boundary
        }

        let cell = self.grid[(ny * self.grid_size as i32 + nx) as usize];
        match cell {
            1 => -0.1,  // wall -- don't move
            3 => {       // hazard -- move but penalty
                self.agent_x = nx as u32;
                self.agent_y = ny as u32;
                -1.0
            }
            _ => {
                self.agent_x = nx as u32;
                self.agent_y = ny as u32;
                if self.agent_x == self.target_x && self.agent_y == self.target_y {
                    10.0 // reached target!
                } else {
                    -0.01 // small step cost
                }
            }
        }
    }

    pub fn agent_x(&self) -> u32 { self.agent_x }
    pub fn agent_y(&self) -> u32 { self.agent_y }
    pub fn target_x(&self) -> u32 { self.target_x }
    pub fn target_y(&self) -> u32 { self.target_y }
    pub fn grid_size_val(&self) -> u32 { self.grid_size }
    pub fn get_tick(&self) -> u64 { self.tick_count }
    pub fn get_total_reward(&self) -> f64 { self.total_reward }
    pub fn get_prediction_error(&self) -> f64 { self.prediction_error }
    pub fn get_current_action(&self) -> u32 { self.current_action }

    /// Get grid data as flat array
    pub fn grid_data(&self) -> Vec<u8> { self.grid.clone() }

    /// Get module activity levels (6 values)
    pub fn module_activities(&self) -> Vec<f32> {
        vec![
            self.sensory.snapshot().activity_level,
            self.associative.snapshot().activity_level,
            self.predictive.snapshot().activity_level,
            self.episodic.snapshot().activity_level,
            self.action_selector.snapshot().activity_level,
            self.safety.snapshot().activity_level,
        ]
    }

    /// Reset the simulation
    pub fn reset(&mut self) {
        self.sensory.reset();
        self.predictive.reset();
        self.action_selector.reset();
        self.episodic.reset();
        self.safety.reset();
        self.agent_x = 1;
        self.agent_y = 1;
        self.total_reward = 0.0;
        self.tick_count = 0;
        self.sim_time = 0.0;
        self.spike_buffer.clear();
    }
}
