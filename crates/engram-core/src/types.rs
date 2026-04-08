use serde::{Deserialize, Serialize};

/// Unique neuron identifier within a module
pub type NeuronId = u32;
/// Synaptic weight
pub type Weight = f32;
/// Simulation time in milliseconds
pub type SimTime = f64;

/// Identifies a brain module/region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum ModuleId {
    Sensory = 0,
    AssociativeMemory = 1,
    PredictiveError = 2,
    EpisodicMemory = 3,
    ActionSelector = 4,
    SafetyKernel = 5,
}

impl ModuleId {
    pub fn name(&self) -> &'static str {
        match self {
            ModuleId::Sensory => "Sensory Cortex",
            ModuleId::AssociativeMemory => "Associative Memory",
            ModuleId::PredictiveError => "Predictive Error",
            ModuleId::EpisodicMemory => "Episodic Memory",
            ModuleId::ActionSelector => "Action Selector",
            ModuleId::SafetyKernel => "Safety Kernel",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            ModuleId::Sensory => "#00d4ff",
            ModuleId::AssociativeMemory => "#a855f7",
            ModuleId::PredictiveError => "#ff8c00",
            ModuleId::EpisodicMemory => "#22d3ee",
            ModuleId::ActionSelector => "#22c55e",
            ModuleId::SafetyKernel => "#ef4444",
        }
    }

    pub fn all() -> &'static [ModuleId] {
        &[
            ModuleId::Sensory,
            ModuleId::AssociativeMemory,
            ModuleId::PredictiveError,
            ModuleId::EpisodicMemory,
            ModuleId::ActionSelector,
            ModuleId::SafetyKernel,
        ]
    }
}

/// A single spike event flowing through the event bus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub source_module: ModuleId,
    pub neuron_id: NeuronId,
    pub timestamp: SimTime,
    pub strength: f32, // normalized [0, 1]
}

/// A proposed action from the action selector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    pub action_id: u32,
    pub confidence: f32,
    pub is_reflex: bool,
    pub timestamp: SimTime,
}

/// A veto event from the safety kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VetoEvent {
    pub timestamp: SimTime,
    pub vetoed_action: ProposedAction,
    pub reason: VetoReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VetoReason {
    HardConstraint(String),
    LearnedInhibition { confidence: f32 },
    EnergyBudgetExceeded,
}

/// Snapshot of a single module for dashboard visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSnapshot {
    pub module_id: ModuleId,
    pub neuron_count: u32,
    pub active_count: u32,
    pub avg_potential: f32,
    pub spike_rate: f32,    // spikes per second
    pub activity_level: f32, // normalized [0, 1]
}

/// Memory formation event for dashboard heatmap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFormation {
    pub timestamp: SimTime,
    pub location_id: u32,
    pub strength: f32,
    pub memory_type: MemoryType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Associative,
    Episodic,
    Consolidated,
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub tick: u64,
    pub sim_time: SimTime,
    pub ticks_per_second: f64,
    pub total_spikes: u64,
    pub total_vetoes: u64,
    pub active_synapses: u64,
    pub memory_bytes: u64,
    pub energy_units: f64,
}

/// Full runtime snapshot sent to dashboard at 30fps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub metrics: RuntimeMetrics,
    pub modules: Vec<ModuleSnapshot>,
    pub recent_spikes: Vec<SpikeEvent>,
    pub recent_vetoes: Vec<VetoEvent>,
    pub prediction_error: f64,
    pub memory_formations: Vec<MemoryFormation>,
    pub current_action: Option<u32>,
    pub total_reward: f64,
    pub agent_position: Option<(u32, u32)>,
}

/// Wire protocol messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    Hello {
        version: String,
        module_names: Vec<String>,
        module_colors: Vec<String>,
    },
    Snapshot(RuntimeSnapshot),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    Start,
    Pause,
    Reset,
    Step,
    SetSpeed(f64),
    WorldEdit { x: u32, y: u32, cell_type: u8 },
}
