use serde::{Deserialize, Serialize};
use std::io;

/// Serialize a value to MessagePack bytes
pub fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec(value)
}

/// Deserialize from MessagePack bytes
pub fn deserialize<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

/// Serialize to JSON string (for Python API and debugging)
pub fn to_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string(value)
}

/// Save bytes to a file
pub fn save_to_file(path: &str, data: &[u8]) -> io::Result<()> {
    std::fs::write(path, data)
}

/// Load bytes from a file
pub fn load_from_file(path: &str) -> io::Result<Vec<u8>> {
    std::fs::read(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RuntimeMetrics;

    #[test]
    fn roundtrip_msgpack() {
        let metrics = RuntimeMetrics {
            tick: 42,
            sim_time: 42.0,
            ticks_per_second: 1000.0,
            total_spikes: 1234,
            total_vetoes: 5,
            active_synapses: 50000,
            memory_bytes: 1024 * 1024,
            energy_units: 3.14,
        };
        let bytes = serialize(&metrics).unwrap();
        let restored: RuntimeMetrics = deserialize(&bytes).unwrap();
        assert_eq!(restored.tick, 42);
        assert_eq!(restored.total_spikes, 1234);
    }
}
