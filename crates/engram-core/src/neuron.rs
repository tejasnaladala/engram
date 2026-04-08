use serde::{Deserialize, Serialize};
use crate::types::SimTime;

/// Parameters for a Leaky Integrate-and-Fire neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFParams {
    /// Membrane time constant (ms)
    pub tau_m: f64,
    /// Resting membrane potential (mV)
    pub v_rest: f64,
    /// Spike threshold potential (mV)
    pub v_threshold: f64,
    /// Reset potential after spike (mV)
    pub v_reset: f64,
    /// Membrane resistance (MOhm)
    pub r_membrane: f64,
    /// Absolute refractory period (ms)
    pub refractory_ms: f64,
}

impl Default for LIFParams {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_rest: -65.0,
            v_threshold: -55.0,
            v_reset: -70.0,
            r_membrane: 10.0,
            refractory_ms: 2.0,
        }
    }
}

/// A single Leaky Integrate-and-Fire neuron with runtime state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LIFNeuron {
    pub params: LIFParams,
    /// Current membrane potential (mV)
    pub potential: f64,
    /// Remaining refractory time (ms)
    pub refractory_remaining: f64,
    /// Time of last spike (ms)
    pub last_spike_time: SimTime,
    /// Accumulated input current for this timestep
    pub input_current: f64,
    /// Total spike count (lifetime)
    pub spike_count: u64,
}

impl LIFNeuron {
    pub fn new(params: LIFParams) -> Self {
        let v_rest = params.v_rest;
        Self {
            params,
            potential: v_rest,
            refractory_remaining: 0.0,
            last_spike_time: -1000.0, // far in the past
            input_current: 0.0,
            spike_count: 0,
        }
    }

    /// Advance the neuron by one timestep. Returns true if the neuron spiked.
    ///
    /// Uses forward Euler integration:
    ///   dV/dt = (-(V - V_rest) + R * I) / tau_m
    ///   V(t+dt) = V(t) + dt * dV/dt
    pub fn step(&mut self, dt: f64, sim_time: SimTime) -> bool {
        // During refractory period: hold at reset, count down
        if self.refractory_remaining > 0.0 {
            self.refractory_remaining -= dt;
            self.potential = self.params.v_reset;
            self.input_current = 0.0;
            return false;
        }

        // Forward Euler integration of membrane potential
        let dv = (dt / self.params.tau_m)
            * (-(self.potential - self.params.v_rest)
                + self.params.r_membrane * self.input_current);
        self.potential += dv;

        // Clear input accumulator
        self.input_current = 0.0;

        // Check for spike
        if self.potential >= self.params.v_threshold {
            self.potential = self.params.v_reset;
            self.refractory_remaining = self.params.refractory_ms;
            self.last_spike_time = sim_time;
            self.spike_count += 1;
            return true;
        }

        false
    }

    /// Accumulate synaptic input current
    #[inline]
    pub fn receive_input(&mut self, current: f64) {
        self.input_current += current;
    }

    /// Reset neuron to initial state
    pub fn reset(&mut self) {
        self.potential = self.params.v_rest;
        self.refractory_remaining = 0.0;
        self.last_spike_time = -1000.0;
        self.input_current = 0.0;
        self.spike_count = 0;
    }
}

/// A population of LIF neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronPopulation {
    pub neurons: Vec<LIFNeuron>,
}

impl NeuronPopulation {
    /// Create a population of identical neurons
    pub fn new(count: usize, params: LIFParams) -> Self {
        Self {
            neurons: (0..count).map(|_| LIFNeuron::new(params.clone())).collect(),
        }
    }

    /// Step all neurons, return indices of those that spiked
    pub fn step(&mut self, dt: f64, sim_time: SimTime) -> Vec<u32> {
        let mut spiked = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if neuron.step(dt, sim_time) {
                spiked.push(i as u32);
            }
        }
        spiked
    }

    /// Deliver input current to a specific neuron
    #[inline]
    pub fn deliver_input(&mut self, neuron_id: u32, current: f64) {
        if let Some(n) = self.neurons.get_mut(neuron_id as usize) {
            n.receive_input(current);
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Count neurons currently above a given potential threshold
    pub fn active_count(&self, threshold_frac: f64) -> u32 {
        let threshold = self.neurons[0].params.v_rest
            + threshold_frac * (self.neurons[0].params.v_threshold - self.neurons[0].params.v_rest);
        self.neurons
            .iter()
            .filter(|n| n.potential > threshold)
            .count() as u32
    }

    /// Average membrane potential across all neurons
    pub fn avg_potential(&self) -> f32 {
        if self.neurons.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.neurons.iter().map(|n| n.potential).sum();
        (sum / self.neurons.len() as f64) as f32
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neuron_spikes_with_sufficient_input() {
        let mut neuron = LIFNeuron::new(LIFParams::default());
        // Inject strong current
        neuron.receive_input(5.0);
        // Step multiple times until spike
        let mut spiked = false;
        for i in 0..100 {
            if neuron.step(1.0, i as f64) {
                spiked = true;
                break;
            }
            neuron.receive_input(5.0);
        }
        assert!(spiked, "Neuron should spike with sustained strong input");
        assert!(neuron.spike_count > 0);
    }

    #[test]
    fn neuron_respects_refractory_period() {
        let mut neuron = LIFNeuron::new(LIFParams {
            refractory_ms: 5.0,
            ..LIFParams::default()
        });
        // Force a spike
        neuron.potential = -54.0; // just below threshold
        neuron.receive_input(10.0);
        assert!(neuron.step(1.0, 0.0));
        // During refractory, should not spike even with strong input
        neuron.receive_input(100.0);
        assert!(!neuron.step(1.0, 1.0));
        assert!(!neuron.step(1.0, 2.0));
    }

    #[test]
    fn neuron_decays_without_input() {
        let mut neuron = LIFNeuron::new(LIFParams::default());
        neuron.potential = -60.0; // above rest
        neuron.step(1.0, 0.0);
        // Should decay toward v_rest (-65)
        assert!(neuron.potential < -60.0);
        assert!(neuron.potential > -65.0);
    }

    #[test]
    fn population_step() {
        let mut pop = NeuronPopulation::new(10, LIFParams::default());
        // Give strong input to neuron 3
        for _ in 0..50 {
            pop.deliver_input(3, 5.0);
            let spiked = pop.step(1.0, 0.0);
            if spiked.contains(&3) {
                return; // success
            }
        }
        panic!("Neuron 3 should have spiked");
    }
}
