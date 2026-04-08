use std::sync::Arc;
use axum::extract::ws::{Message, WebSocket};
use tokio::sync::{broadcast, Mutex};
use futures_util::{SinkExt, StreamExt};

use engram_core::ClientMessage;
use engram_runtime::EngramRuntime;

use crate::protocol::{decode_client_message, encode_server_message, hello_message};

/// Handle a single WebSocket connection
pub async fn handle_socket(
    socket: WebSocket,
    runtime: Arc<Mutex<EngramRuntime>>,
    mut rx: broadcast::Receiver<Vec<u8>>,
) {
    let (mut sender, mut receiver) = socket.split();

    // Send Hello message
    let hello = encode_server_message(&hello_message());
    if sender.send(Message::Binary(hello.into())).await.is_err() {
        return;
    }

    // Receive task: handle client commands
    let rt_clone = runtime.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Binary(data) = msg {
                if let Some(cmd) = decode_client_message(&data) {
                    let mut rt = rt_clone.lock().await;
                    match cmd {
                        ClientMessage::Start => rt.running = true,
                        ClientMessage::Pause => rt.running = false,
                        ClientMessage::Reset => rt.full_reset(),
                        ClientMessage::Step => { rt.step(); },
                        ClientMessage::SetSpeed(_speed) => {
                            // Speed handled at broadcaster level
                        },
                        ClientMessage::WorldEdit { .. } => {
                            // Grid editing handled by environment layer
                        },
                    }
                }
            }
        }
    });

    // Send task: forward broadcast snapshots to this client
    let send_task = tokio::spawn(async move {
        while let Ok(data) = rx.recv().await {
            if sender.send(Message::Binary(data.into())).await.is_err() {
                break;
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }
}
