wit_bindgen::generate!();

mod errors;
mod streams;
mod http_handler;
mod nats_handler;

pub use exports::wasi::http::incoming_handler::Guest as HttpGuest;
pub use exports::wasmcloud::messaging::handler::Guest as NatsGuest;

// pub use exports::wasmcloud::messaging::types::*;
// pub use exports::wasmcloud::messaging::consumer;

pub struct Api;

export!(Api);