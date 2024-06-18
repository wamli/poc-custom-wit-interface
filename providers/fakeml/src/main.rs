use anyhow::Context;
// use tracing::{debug, info, error};
use wasmcloud_provider_sdk::{run_provider, load_host_data, get_connection};

mod config;
mod provider;

use provider::AiModelProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	AiModelProvider::run().await?;
	eprintln!("AI model provider exiting");
	Ok(())
}