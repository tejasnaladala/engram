use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, SpikeEvent, SimTime,
    MemoryFormation, MemoryType,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Sparse Distributed Memory (SDM) based associative memory.
///
/// Uses Kanerva's model: N hard locations with random addresses.
/// Patterns are stored by incrementing/decrementing counters at
/// nearby locations. Retrieval uses superposition of counter vectors
/// from nearby locations. Naturally resistant to catastrophic forgetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociativeMemory {
    pub population: NeuronPopulation,
    /// Number of hard locations
    num_locations: usize,
    /// Data width (bits per pattern)
    data_width: usize,
    /// Random binary addresses for hard locations (flattened: num_locations * data_width)
    addresses: Vec<bool>,
    /// Counter storage (flattened: num_locations * data_width)
    counters: Vec<i32>,
    /// Access radius (max Hamming distance for activation)
    access_radius: usize,
    /// Current input pattern (accumulated from incoming spikes)
    input_pattern: Vec<f64>,
    /// Last retrieved pattern
    output_pattern: Vec<f64>,
    /// Recent memory formations for dashboard
    recent_formations: Vec<MemoryFormation>,
    /// Write counter
    write_count: u64,
    recent_spike_count: u32,
    seed: u64,
    #[serde(skip, default = "default_rng")]
    rng: ChaCha8Rng,
}

fn default_rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

impl AssociativeMemory {
    pub fn new(num_neurons: usize, num_locations: usize, data_width: usize, seed: u64) -> Self {
        let params = LIFParams {
            tau_m: 25.0, // slower integration -- memory should be persistent
            v_rest: -65.0,
            v_threshold: -52.0,
            v_reset: -68.0,
            r_membrane: 12.0,
            refractory_ms: 2.0,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random binary addresses for each hard location
        let total_bits = num_locations * data_width;
        let addresses: Vec<bool> = (0..total_bits).map(|_| rng.random::<bool>()).collect();
        let counters = vec![0i32; total_bits];

        Self {
            population: NeuronPopulation::new(num_neurons, params),
            num_locations,
            data_width,
            addresses,
            counters,
            access_radius: data_width / 4, // ~25% mismatch tolerance
            input_pattern: vec![0.0; data_width],
            output_pattern: vec![0.0; data_width],
            recent_formations: Vec::new(),
            write_count: 0,
            recent_spike_count: 0,
            seed,
            rng,
        }
    }

    /// Convert spike events to a binary pattern
    fn spikes_to_pattern(&self, spikes: &[SpikeEvent]) -> Vec<bool> {
        let mut pattern = vec![false; self.data_width];
        for spike in spikes {
            let idx = spike.neuron_id as usize % self.data_width;
            pattern[idx] = true;
        }
        pattern
    }

    /// Compute Hamming distance between a pattern and a hard location's address
    fn hamming_distance(&self, pattern: &[bool], location_idx: usize) -> usize {
        let base = location_idx * self.data_width;
        pattern
            .iter()
            .enumerate()
            .filter(|(i, &bit)| bit != self.addresses[base + i])
            .count()
    }

    /// Write a pattern to memory (increment/decrement counters at nearby locations)
    fn write_pattern(&mut self, pattern: &[bool], sim_time: SimTime) {
        for loc in 0..self.num_locations {
            let dist = self.hamming_distance(pattern, loc);
            if dist <= self.access_radius {
                let base = loc * self.data_width;
                for (i, &bit) in pattern.iter().enumerate() {
                    if bit {
                        self.counters[base + i] += 1;
                    } else {
                        self.counters[base + i] -= 1;
                    }
                }
                self.recent_formations.push(MemoryFormation {
                    timestamp: sim_time,
                    location_id: loc as u32,
                    strength: 1.0 - (dist as f32 / self.access_radius as f32),
                    memory_type: MemoryType::Associative,
                });
            }
        }
        self.write_count += 1;
    }

    /// Read from memory: sum counter vectors at nearby locations, threshold at 0
    fn read_pattern(&self, query: &[bool]) -> Vec<bool> {
        let mut sum = vec![0i64; self.data_width];
        for loc in 0..self.num_locations {
            let dist = self.hamming_distance(query, loc);
            if dist <= self.access_radius {
                let base = loc * self.data_width;
                for i in 0..self.data_width {
                    sum[i] += self.counters[base + i] as i64;
                }
            }
        }
        sum.iter().map(|&s| s > 0).collect()
    }

    /// Take and clear recent memory formations (for dashboard)
    pub fn take_formations(&mut self) -> Vec<MemoryFormation> {
        std::mem::take(&mut self.recent_formations)
    }
}

impl BrainModule for AssociativeMemory {
    fn id(&self) -> ModuleId {
        ModuleId::AssociativeMemory
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        // Convert incoming spikes to a binary pattern
        let input_pattern = self.spikes_to_pattern(incoming);

        // Write the current pattern to memory
        if incoming.len() > 2 {
            self.write_pattern(&input_pattern, sim_time);
        }

        // Read from memory using the input as a query (pattern completion)
        let retrieved = self.read_pattern(&input_pattern);

        // Convert retrieved pattern to neural input
        // Update the accumulator
        for (i, &bit) in retrieved.iter().enumerate() {
            let idx = i % self.population.len();
            if bit {
                self.population.deliver_input(idx as u32, 5.0);
                self.output_pattern[i] = 1.0;
            } else {
                self.output_pattern[i] *= 0.95; // decay
            }
        }

        // Also drive neurons from incoming spikes directly
        for spike in incoming {
            let idx = spike.neuron_id as usize % self.population.len();
            self.population.deliver_input(idx as u32, spike.strength as f64 * 3.0);
        }

        // Step neurons
        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        spiked
            .into_iter()
            .map(|nid| SpikeEvent {
                source_module: ModuleId::AssociativeMemory,
                neuron_id: nid,
                timestamp: sim_time,
                strength: 1.0,
            })
            .collect()
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.population.len() as u32;
        ModuleSnapshot {
            module_id: ModuleId::AssociativeMemory,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0,
            activity_level: (self.recent_spike_count as f32 / n as f32).min(1.0),
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.input_pattern.fill(0.0);
        self.output_pattern.fill(0.0);
        self.recent_formations.clear();
        self.recent_spike_count = 0;
        // Note: we do NOT reset counters -- memory persists across episodes
    }

    fn neuron_count(&self) -> u32 {
        self.population.len() as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<AssociativeMemory>(data) {
            *self = restored;
        }
    }
}
