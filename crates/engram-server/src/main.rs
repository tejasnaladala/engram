use std::sync::Arc;
use axum::{extract::ws::WebSocketUpgrade, extract::State, response::IntoResponse, routing::get, Router};
use tokio::sync::{broadcast, Mutex};
use tower_http::cors::CorsLayer;

use engram_runtime::{EngramRuntime, RuntimeConfig};

mod protocol;
mod ws_handler;
mod state_broadcaster;

struct AppState {
    runtime: Arc<Mutex<EngramRuntime>>,
    tx: broadcast::Sender<Vec<u8>>,
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let runtime = state.runtime.clone();
    let rx = state.tx.subscribe();
    ws.on_upgrade(move |socket| ws_handler::handle_socket(socket, runtime, rx))
}

async fn health() -> &'static str {
    "engram-server running"
}

#[tokio::main]
async fn main() {
    let config = RuntimeConfig::default();
    let port = config.ws_port;
    let fps = config.dashboard_fps;

    let runtime = Arc::new(Mutex::new(EngramRuntime::new(config)));
    let (tx, _rx) = broadcast::channel::<Vec<u8>>(16);

    // Start the simulation broadcaster
    let rt_clone = runtime.clone();
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        state_broadcaster::run_broadcaster(rt_clone, tx_clone, fps, 10).await;
    });

    let state = Arc::new(AppState {
        runtime,
        tx,
    });

    let app = Router::new()
        .route("/ws", get(ws_upgrade))
        .route("/health", get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .expect("Failed to bind");

    println!("Engram server listening on port {}", port);
    println!("Dashboard: connect WebSocket to ws://localhost:{}/ws", port);

    axum::serve(listener, app).await.unwrap();
}
