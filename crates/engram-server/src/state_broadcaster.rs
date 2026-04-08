use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tokio::time::{interval, Duration};

use engram_core::ServerMessage;
use engram_runtime::EngramRuntime;

use crate::protocol::encode_server_message;

/// Runs the simulation loop and broadcasts snapshots to connected clients
pub async fn run_broadcaster(
    runtime: Arc<Mutex<EngramRuntime>>,
    tx: broadcast::Sender<Vec<u8>>,
    fps: u32,
    ticks_per_frame: u32,
) {
    let frame_duration = Duration::from_millis(1000 / fps as u64);
    let mut ticker = interval(frame_duration);

    loop {
        ticker.tick().await;

        let snapshot_bytes = {
            let mut rt = runtime.lock().await;
            if !rt.running {
                // Still send snapshots even when paused (for connection keepalive)
                let snapshot = rt.snapshot();
                encode_server_message(&ServerMessage::Snapshot(snapshot))
            } else {
                // Run simulation ticks
                for _ in 0..ticks_per_frame {
                    rt.step();
                }
                let snapshot = rt.snapshot();
                encode_server_message(&ServerMessage::Snapshot(snapshot))
            }
        };

        // Broadcast to all connected clients (ignore send errors — no clients connected)
        let _ = tx.send(snapshot_bytes);
    }
}
