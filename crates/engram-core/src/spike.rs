use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use crate::types::{SpikeEvent, SimTime};

/// Ring buffer for recent spike events with automatic eviction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeBuffer {
    buffer: VecDeque<SpikeEvent>,
    capacity: usize,
}

impl SpikeBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a spike event, evicting the oldest if at capacity
    pub fn push(&mut self, event: SpikeEvent) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(event);
    }

    /// Add multiple spike events
    pub fn extend(&mut self, events: impl IntoIterator<Item = SpikeEvent>) {
        for event in events {
            self.push(event);
        }
    }

    /// Get all spikes within a time window ending at current_time
    pub fn recent(&self, window_ms: SimTime, current_time: SimTime) -> Vec<&SpikeEvent> {
        let cutoff = current_time - window_ms;
        self.buffer
            .iter()
            .filter(|e| e.timestamp >= cutoff)
            .collect()
    }

    /// Get all spikes in the buffer
    pub fn all(&self) -> impl Iterator<Item = &SpikeEvent> {
        self.buffer.iter()
    }

    /// Drain and return all events, clearing the buffer
    pub fn drain(&mut self) -> Vec<SpikeEvent> {
        self.buffer.drain(..).collect()
    }

    /// Current count
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Record of spikes for a single neuron over time (for analysis/benchmarking)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTrain {
    pub neuron_id: u32,
    pub times: Vec<SimTime>,
}

impl SpikeTrain {
    pub fn new(neuron_id: u32) -> Self {
        Self {
            neuron_id,
            times: Vec::new(),
        }
    }

    pub fn record(&mut self, time: SimTime) {
        self.times.push(time);
    }

    /// Compute mean firing rate over a window (Hz)
    pub fn firing_rate(&self, start: SimTime, end: SimTime) -> f64 {
        let duration_s = (end - start) / 1000.0;
        if duration_s <= 0.0 {
            return 0.0;
        }
        let count = self
            .times
            .iter()
            .filter(|&&t| t >= start && t < end)
            .count();
        count as f64 / duration_s
    }

    /// Compute inter-spike intervals (ms)
    pub fn isi(&self) -> Vec<f64> {
        self.times.windows(2).map(|w| w[1] - w[0]).collect()
    }

    pub fn spike_count(&self) -> usize {
        self.times.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModuleId;

    fn make_spike(t: SimTime) -> SpikeEvent {
        SpikeEvent {
            source_module: ModuleId::Sensory,
            neuron_id: 0,
            timestamp: t,
            strength: 1.0,
        }
    }

    #[test]
    fn spike_buffer_evicts_oldest() {
        let mut buf = SpikeBuffer::new(3);
        buf.push(make_spike(1.0));
        buf.push(make_spike(2.0));
        buf.push(make_spike(3.0));
        buf.push(make_spike(4.0));
        assert_eq!(buf.len(), 3);
        let all: Vec<_> = buf.all().collect();
        assert!((all[0].timestamp - 2.0).abs() < 1e-6);
    }

    #[test]
    fn spike_buffer_recent_window() {
        let mut buf = SpikeBuffer::new(100);
        for t in 0..50 {
            buf.push(make_spike(t as f64));
        }
        let recent = buf.recent(10.0, 49.0);
        // Spikes with timestamp >= 39.0: t=39,40,41,...,49 = 11 spikes
        assert_eq!(recent.len(), 11);
    }

    #[test]
    fn spike_train_firing_rate() {
        let mut train = SpikeTrain::new(0);
        for t in (0..1000).step_by(10) {
            train.record(t as f64);
        }
        let rate = train.firing_rate(0.0, 1000.0);
        assert!((rate - 100.0).abs() < 1.0, "Expected ~100 Hz, got {}", rate);
    }
}
