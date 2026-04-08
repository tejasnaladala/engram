use engram_core::{
    BrainModule, LIFParams, ModuleId, ModuleSnapshot, NeuronPopulation, SpikeEvent, SimTime,
    MemoryFormation, MemoryType,
};
use serde::{Deserialize, Serialize};

/// A single frame of experience within an episode
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpisodeFrame {
    timestamp: SimTime,
    spike_pattern: Vec<u32>, // neuron IDs that were active
    reward: f64,
    prediction_error: f64,
}

/// A complete episode (sequence of frames)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Episode {
    frames: Vec<EpisodeFrame>,
    importance: f64,
    total_reward: f64,
    max_error: f64,
}

/// Episodic Memory: records sequences of experience, scores them by importance,
/// and replays the most important episodes during idle periods for consolidation.
///
/// This implements a prioritized experience replay mechanism inspired by
/// hippocampal replay during sleep. Important episodes (high reward or high
/// surprise) are replayed more frequently, strengthening their representation
/// in associative memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub population: NeuronPopulation,
    /// Stored episodes (ring buffer)
    episodes: Vec<Episode>,
    /// Current episode being recorded
    current_episode: Episode,
    /// Maximum number of stored episodes
    max_episodes: usize,
    /// Maximum frames per episode
    max_frames: usize,
    /// Replay interval (every N ticks)
    replay_interval: u64,
    /// Ticks since last replay
    ticks_since_replay: u64,
    /// Whether currently replaying
    is_replaying: bool,
    /// Current replay frame index
    replay_frame_idx: usize,
    /// Index of episode being replayed
    replay_episode_idx: usize,
    /// Recent memory formations for dashboard
    recent_formations: Vec<MemoryFormation>,
    recent_spike_count: u32,
}

impl EpisodicMemory {
    pub fn new(num_neurons: usize, max_episodes: usize) -> Self {
        let params = LIFParams {
            tau_m: 30.0, // slow -- episodic memory is persistent
            v_rest: -65.0,
            v_threshold: -53.0,
            v_reset: -68.0,
            r_membrane: 10.0,
            refractory_ms: 3.0,
        };

        Self {
            population: NeuronPopulation::new(num_neurons, params),
            episodes: Vec::new(),
            current_episode: Episode {
                frames: Vec::new(),
                importance: 0.0,
                total_reward: 0.0,
                max_error: 0.0,
            },
            max_episodes,
            max_frames: 500,
            replay_interval: 1000,
            ticks_since_replay: 0,
            is_replaying: false,
            replay_frame_idx: 0,
            replay_episode_idx: 0,
            recent_formations: Vec::new(),
            recent_spike_count: 0,
        }
    }

    /// Record a frame of experience
    pub fn record_frame(&mut self, spikes: &[SpikeEvent], reward: f64, prediction_error: f64) {
        let pattern: Vec<u32> = spikes.iter().map(|s| s.neuron_id).collect();
        self.current_episode.frames.push(EpisodeFrame {
            timestamp: spikes.first().map(|s| s.timestamp).unwrap_or(0.0),
            spike_pattern: pattern,
            reward,
            prediction_error,
        });
        self.current_episode.total_reward += reward;
        if prediction_error > self.current_episode.max_error {
            self.current_episode.max_error = prediction_error;
        }
    }

    /// End the current episode and store it
    pub fn end_episode(&mut self) {
        if self.current_episode.frames.is_empty() {
            return;
        }

        // Score importance: |total_reward| + max(prediction_error)
        self.current_episode.importance =
            self.current_episode.total_reward.abs() + self.current_episode.max_error;

        let episode = std::mem::replace(
            &mut self.current_episode,
            Episode {
                frames: Vec::new(),
                importance: 0.0,
                total_reward: 0.0,
                max_error: 0.0,
            },
        );

        if self.episodes.len() >= self.max_episodes {
            // Remove least important episode
            if let Some(min_idx) = self
                .episodes
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.importance.partial_cmp(&b.1.importance).unwrap())
                .map(|(i, _)| i)
            {
                if episode.importance > self.episodes[min_idx].importance {
                    self.episodes[min_idx] = episode;
                }
            }
        } else {
            self.episodes.push(episode);
        }
    }

    /// Get the most important episode for replay
    fn most_important_episode(&self) -> Option<usize> {
        self.episodes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.importance.partial_cmp(&b.1.importance).unwrap())
            .map(|(i, _)| i)
    }

    /// Take and clear recent formations
    pub fn take_formations(&mut self) -> Vec<MemoryFormation> {
        std::mem::take(&mut self.recent_formations)
    }
}

impl BrainModule for EpisodicMemory {
    fn id(&self) -> ModuleId {
        ModuleId::EpisodicMemory
    }

    fn step(&mut self, dt: f64, sim_time: SimTime, incoming: &[SpikeEvent]) -> Vec<SpikeEvent> {
        self.ticks_since_replay += 1;
        let mut output_spikes = Vec::new();

        // Handle replay
        if self.is_replaying {
            if let Some(episode) = self.episodes.get(self.replay_episode_idx) {
                if self.replay_frame_idx < episode.frames.len() {
                    let frame = &episode.frames[self.replay_frame_idx];
                    // Inject replay spikes
                    for &nid in &frame.spike_pattern {
                        let idx = nid as usize % self.population.len();
                        self.population.deliver_input(idx as u32, 8.0);
                    }
                    self.replay_frame_idx += 1;

                    self.recent_formations.push(MemoryFormation {
                        timestamp: sim_time,
                        location_id: self.replay_episode_idx as u32,
                        strength: episode.importance as f32,
                        memory_type: MemoryType::Consolidated,
                    });
                } else {
                    // Replay complete
                    self.is_replaying = false;
                    self.replay_frame_idx = 0;
                }
            } else {
                self.is_replaying = false;
            }
        } else if self.ticks_since_replay >= self.replay_interval && !self.episodes.is_empty() {
            // Start replay of most important episode
            if let Some(idx) = self.most_important_episode() {
                self.is_replaying = true;
                self.replay_episode_idx = idx;
                self.replay_frame_idx = 0;
                self.ticks_since_replay = 0;
            }
        }

        // Pass through incoming spikes to neurons
        for spike in incoming {
            let idx = spike.neuron_id as usize % self.population.len();
            self.population.deliver_input(idx as u32, spike.strength as f64 * 2.0);
        }

        // Step neurons
        let spiked = self.population.step(dt, sim_time);
        self.recent_spike_count = spiked.len() as u32;

        for nid in spiked {
            output_spikes.push(SpikeEvent {
                source_module: ModuleId::EpisodicMemory,
                neuron_id: nid,
                timestamp: sim_time,
                strength: if self.is_replaying { 0.7 } else { 1.0 },
            });
        }

        output_spikes
    }

    fn snapshot(&self) -> ModuleSnapshot {
        let n = self.population.len() as u32;
        ModuleSnapshot {
            module_id: ModuleId::EpisodicMemory,
            neuron_count: n,
            active_count: self.recent_spike_count,
            avg_potential: self.population.avg_potential(),
            spike_rate: self.recent_spike_count as f32 * 1000.0,
            activity_level: if self.is_replaying {
                1.0
            } else {
                (self.recent_spike_count as f32 / n as f32).min(1.0)
            },
        }
    }

    fn reset(&mut self) {
        self.population.reset();
        self.end_episode(); // save current episode before reset
        self.is_replaying = false;
        self.ticks_since_replay = 0;
        self.recent_spike_count = 0;
    }

    fn neuron_count(&self) -> u32 {
        self.population.len() as u32
    }

    fn save_state(&self) -> Vec<u8> {
        engram_core::checkpoint::serialize(self).unwrap_or_default()
    }

    fn load_state(&mut self, data: &[u8]) {
        if let Ok(restored) = engram_core::checkpoint::deserialize::<EpisodicMemory>(data) {
            *self = restored;
        }
    }
}
