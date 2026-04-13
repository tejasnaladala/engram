// Engram Design System -- Muted Clinical Palette
// Monochrome-ish teal/slate/blue tones. NOT rainbow.

/** Module colors -- muted, cohesive, not saturated */
export const MODULE_COLORS = {
  Sensory: '#5098a8',
  AssociativeMemory: '#5878a0',
  PredictiveError: '#907858',
  EpisodicMemory: '#508898',
  ActionSelector: '#508870',
  SafetyKernel: '#906060',
} as const

export const MODULE_NAMES = [
  'Somatosensory Ctx.',
  'Association Ctx.',
  'Pred. Error Signal',
  'Hippocampal Mem.',
  'Motor Cortex',
  'Brainstem Gov.',
]

export const MODULE_ABBREVS = ['S1', 'ASC', 'PE', 'HPC', 'M1', 'BST']

export const MODULE_COLOR_ARRAY = [
  '#5098a8',
  '#5878a0',
  '#907858',
  '#508898',
  '#508870',
  '#906060',
]

export const MODULE_COLOR_DIM = [
  'rgba(80, 152, 168, 0.10)',
  'rgba(88, 120, 160, 0.10)',
  'rgba(144, 120, 88, 0.10)',
  'rgba(80, 136, 152, 0.10)',
  'rgba(80, 136, 112, 0.10)',
  'rgba(144, 96, 96, 0.10)',
]

export const SURFACES = {
  s0: '#050508',
  s1: '#0a0b10',
  s2: '#0e1017',
  s3: '#13151e',
  s4: '#191c28',
  s5: '#1f2333',
}

export const ACCENT = {
  primary: '#3098a8',
  primaryDim: 'rgba(48, 152, 168, 0.12)',
  primaryGlow: 'rgba(48, 152, 168, 0.06)',
}

export const MODULE_NEURON_COUNTS = [128, 256, 64, 64, 128, 32]
export const TOTAL_NEURONS = 672
