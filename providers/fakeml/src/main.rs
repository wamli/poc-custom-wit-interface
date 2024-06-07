use anyhow::Context;
// use tracing::{debug, info};
use wasmcloud_provider_sdk::{run_provider, load_host_data, get_connection};
// use wasmcloud_provider_wit_bindgen::deps::wasmcloud_provider_sdk::{start_provider, load_host_data};
// use wasmcloud_provider_sdk::{start_provider, load_host_data};

mod config;
mod provider;

use provider::AiModelProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	AiModelProvider::run().await?;
	eprintln!("AI model provider exiting");
	Ok(())
}

// #[tokio::main]
// async fn main() -> anyhow::Result<()> {
// 	// load host data
// 	let host_data = load_host_data().context("failed to load host data")?;
// 	// initialize provider with host data
// 	let provider = Self::from_host_data(host_data);
// 	// Run the provider itself, initialize links
// 	let shutdown = run_provider(provider.clone(), "messaging-nats-provider")
// 	.await
// 	.context("failed to run provider")?;
// 	// get RPC connection
// 	let connection = get_connection();
// 	// serve RPC
// 	serve(
// 		&connection.get_wrpc_client(connection.provider_key()),
// 		provider,
// 		shutdown,
// 	)
// 	.await;
// 	// exit messsage
// 	eprintln!("provider exiting");
// }