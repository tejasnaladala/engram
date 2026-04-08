// Protocol types are defined in engram-core::types (ServerMessage, ClientMessage)
// This module provides serialization helpers

use engram_core::{ServerMessage, ClientMessage, RuntimeSnapshot, ModuleId};

/// Serialize a server message to MessagePack bytes
pub fn encode_server_message(msg: &ServerMessage) -> Vec<u8> {
    rmp_serde::to_vec(msg).unwrap_or_default()
}

/// Deserialize a client message from MessagePack bytes
pub fn decode_client_message(bytes: &[u8]) -> Option<ClientMessage> {
    rmp_serde::from_slice(bytes).ok()
}

/// Serialize a server message to JSON (fallback/debug)
pub fn encode_json(msg: &ServerMessage) -> String {
    serde_json::to_string(msg).unwrap_or_default()
}

/// Create the initial Hello message
pub fn hello_message() -> ServerMessage {
    ServerMessage::Hello {
        version: env!("CARGO_PKG_VERSION").to_string(),
        module_names: ModuleId::all().iter().map(|m| m.name().to_string()).collect(),
        module_colors: ModuleId::all().iter().map(|m| m.color().to_string()).collect(),
    }
}
