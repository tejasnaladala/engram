// Engram Neuro-Clinical Design System -- Theme Constants
// Based on radiology workstation conventions, neuroimaging color maps,
// and computational neuroscience visualization standards.

/** Module colors -- mapped to neuroanatomical conventions */
export const MODULE_COLORS = {
  Sensory: '#3dd8e0',          // Somatosensory cortex -- cool cyan
  AssociativeMemory: '#8b6cf0', // Association cortex -- violet
  PredictiveError: '#e8943a',   // Predictive coding -- metabolic amber
  EpisodicMemory: '#4ac0d0',   // Hippocampal -- teal
  ActionSelector: '#3dbd5c',   // Motor cortex -- clinical green
  SafetyKernel: '#d94040',     // Brainstem/autonomic -- diagnostic red
} as const

/** Anatomically accurate module naming */
export const MODULE_NAMES = [
  'Somatosensory Ctx.',
  'Association Ctx.',
  'Pred. Error Signal',
  'Hippocampal Mem.',
  'Motor Cortex',
  'Brainstem Gov.',
]

/** Short clinical abbreviations */
export const MODULE_ABBREVS = [
  'S1',    // Primary somatosensory
  'ASC',   // Association cortex
  'PE',    // Prediction error
  'HPC',   // Hippocampus
  'M1',    // Primary motor
  'BST',   // Brainstem
]

export const MODULE_COLOR_ARRAY = [
  '#3dd8e0',
  '#8b6cf0',
  '#e8943a',
  '#4ac0d0',
  '#3dbd5c',
  '#d94040',
]

/** Dim variants for backgrounds / inactive states */
export const MODULE_COLOR_DIM = [
  'rgba(61, 216, 224, 0.12)',
  'rgba(139, 108, 240, 0.12)',
  'rgba(232, 148, 58, 0.12)',
  'rgba(74, 192, 208, 0.12)',
  'rgba(61, 189, 92, 0.12)',
  'rgba(217, 64, 64, 0.12)',
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
  primary: '#3dd8e0',
  primaryDim: 'rgba(61, 216, 224, 0.15)',
  primaryGlow: 'rgba(61, 216, 224, 0.08)',
}

/** Neuron count per module -- for visualization scaling */
export const MODULE_NEURON_COUNTS = [128, 256, 64, 64, 128, 32]
export const TOTAL_NEURONS = 672
