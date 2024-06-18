use std::collections::HashMap;

use serde::{Deserialize, Serialize};


/// Default URL to use to connect to registry
const DEFAULT_CONNECT_URL: &str = "localhost:5000";

/// Configuration key that will be used to search for config url
const CONFIG_URL_KEY: &str = "URL";


/// Configuration for this provider, which is passed to the provider from the host.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderConfig {
    values: HashMap<String, String>,
}

impl From<&HashMap<String, String>> for ProviderConfig {
    /// Construct configuration struct from the passed config values.
    ///
    /// For this example, we just store the values directly for any later reference.
    /// You can use this as a base to create your own strongly typed configuration struct.
    fn from(values: &HashMap<String, String>) -> ProviderConfig {
        ProviderConfig {
            values: values.clone(),
        }
    }
}