use pyo3::prelude::*;
use engram_runtime::{EngramRuntime, RuntimeConfig};

/// Python wrapper for the Engram cognitive runtime
#[pyclass]
struct PyRuntime {
    runtime: EngramRuntime,
}

#[pymethods]
impl PyRuntime {
    #[new]
    #[pyo3(signature = (input_dims=8, num_actions=4, seed=42))]
    fn new(input_dims: usize, num_actions: usize, seed: u64) -> Self {
        let config = RuntimeConfig {
            input_dims,
            num_actions,
            seed,
            ..RuntimeConfig::default()
        };
        Self {
            runtime: EngramRuntime::new(config),
        }
    }

    /// Set the current observation (list of floats in [0, 1])
    fn set_observation(&mut self, obs: Vec<f64>) {
        self.runtime.set_observation(&obs);
    }

    /// Set the reward signal
    fn set_reward(&mut self, reward: f64) {
        self.runtime.set_reward(reward);
    }

    /// Run one cognitive cycle, returns the selected action ID
    fn step(&mut self) -> u32 {
        self.runtime.step()
    }

    /// Get current prediction error
    fn prediction_error(&self) -> f64 {
        self.runtime.prediction_error()
    }

    /// Get current tick count
    fn tick(&self) -> u64 {
        self.runtime.tick()
    }

    /// Get total lifetime spike count
    fn total_spikes(&self) -> u64 {
        self.runtime.total_spikes()
    }

    /// Get total lifetime veto count
    fn total_vetoes(&self) -> u64 {
        self.runtime.total_vetoes()
    }

    /// Reset for a new episode (preserves learned memories)
    fn reset_episode(&mut self) {
        self.runtime.reset_episode();
    }

    /// Full reset (clears everything)
    fn full_reset(&mut self) {
        self.runtime.full_reset();
    }

    /// Get current state as JSON string
    fn snapshot_json(&mut self) -> String {
        let snapshot = self.runtime.snapshot();
        serde_json::to_string(&snapshot).unwrap_or_default()
    }

    /// Get total accumulated reward
    fn total_reward(&self) -> f64 {
        self.runtime.total_reward
    }

    /// Get simulation time in ms
    fn sim_time(&self) -> f64 {
        self.runtime.sim_time
    }
}

/// Python module
#[pymodule]
fn _engram_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    Ok(())
}
