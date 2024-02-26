use wasmcloud_provider_wit_bindgen::deps::wasmcloud_provider_sdk::{start_provider, load_host_data};

use fakeml::AiModelProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hd = load_host_data()?;

    start_provider(
        AiModelProvider::new("DUMMY"),
        Some("wamli-provider".to_string()),
    )?;

    eprintln!("Wamli provider exiting");
    Ok(())
}