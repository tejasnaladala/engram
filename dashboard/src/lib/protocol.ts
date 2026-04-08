// Types matching the Rust RuntimeSnapshot

export interface SpikeEvent {
  source_module: string
  neuron_id: number
  timestamp: number
  strength: number
}

export interface ModuleSnapshot {
  module_id: string
  neuron_count: number
  active_count: number
  avg_potential: number
  spike_rate: number
  activity_level: number
}

export interface VetoEvent {
  timestamp: number
  vetoed_action: {
    action_id: number
    confidence: number
    is_reflex: boolean
  }
  reason: string | { LearnedInhibition: { confidence: number } }
}

export interface MemoryFormation {
  timestamp: number
  location_id: number
  strength: number
  memory_type: string
}

export interface RuntimeMetrics {
  tick: number
  sim_time: number
  ticks_per_second: number
  total_spikes: number
  total_vetoes: number
  active_synapses: number
  memory_bytes: number
  energy_units: number
}

export interface RuntimeSnapshot {
  metrics: RuntimeMetrics
  modules: ModuleSnapshot[]
  recent_spikes: SpikeEvent[]
  recent_vetoes: VetoEvent[]
  prediction_error: number
  memory_formations: MemoryFormation[]
  current_action: number | null
  total_reward: number
  agent_position: [number, number] | null
}

export interface HelloMessage {
  Hello: {
    version: string
    module_names: string[]
    module_colors: string[]
  }
}

// Module index to color mapping
export function getModuleColor(moduleIndex: number): string {
  const colors = ['#00d4ff', '#a855f7', '#ff8c00', '#22d3ee', '#22c55e', '#ef4444']
  return colors[moduleIndex % colors.length]
}

export function getModuleColorByName(name: string): string {
  const map: Record<string, string> = {
    'Sensory': '#00d4ff',
    'AssociativeMemory': '#a855f7',
    'PredictiveError': '#ff8c00',
    'EpisodicMemory': '#22d3ee',
    'ActionSelector': '#22c55e',
    'SafetyKernel': '#ef4444',
  }
  return map[name] || '#888888'
}
