use engram_core::RuntimeMetrics;
use std::time::Instant;

/// Tracks performance metrics for the runtime
pub struct MetricsTracker {
    pub metrics: RuntimeMetrics,
    last_measure: Option<Instant>,
    ticks_since_measure: u64,
    measure_interval_ticks: u64,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            metrics: RuntimeMetrics::default(),
            last_measure: None,
            ticks_since_measure: 0,
            measure_interval_ticks: 100,
        }
    }

    pub fn record_tick(&mut self) {
        self.metrics.tick += 1;
        self.ticks_since_measure += 1;

        // Periodically compute ticks/second
        if self.ticks_since_measure >= self.measure_interval_ticks {
            let now = Instant::now();
            if let Some(last) = self.last_measure {
                let elapsed = now.duration_since(last).as_secs_f64();
                if elapsed > 0.0 {
                    self.metrics.ticks_per_second =
                        self.ticks_since_measure as f64 / elapsed;
                }
            }
            self.last_measure = Some(now);
            self.ticks_since_measure = 0;
        }
    }

    pub fn record_spikes(&mut self, count: u64) {
        self.metrics.total_spikes += count;
    }

    pub fn record_veto(&mut self) {
        self.metrics.total_vetoes += 1;
    }

    pub fn set_sim_time(&mut self, t: f64) {
        self.metrics.sim_time = t;
    }

    pub fn set_active_synapses(&mut self, count: u64) {
        self.metrics.active_synapses = count;
    }

    pub fn add_energy(&mut self, units: f64) {
        self.metrics.energy_units += units;
    }

    pub fn reset(&mut self) {
        self.metrics = RuntimeMetrics::default();
        self.last_measure = None;
        self.ticks_since_measure = 0;
    }
}
