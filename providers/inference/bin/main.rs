// mod config;
// mod provider;
// mod inference;
// mod data_loader;

use inference::provider::InferenceProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    InferenceProvider::run().await?;
    eprintln!("AI model provider exiting");
    Ok(())
}
