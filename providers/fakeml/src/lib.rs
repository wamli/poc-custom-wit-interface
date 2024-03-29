//! Redis implementation for wasmcloud:keyvalue.
//!
//! This implementation is multi-threaded and operations between different actors
use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{error, info, instrument, warn};

// use redis::aio::ConnectionManager;
use wasmtime::component::{Resource, ResourceTable};

use wasmcloud_provider_wit_bindgen::deps::{
    async_trait::async_trait,
    serde::Deserialize,
    serde_json,
    wasmcloud_provider_sdk::core::LinkDefinition,
    wasmcloud_provider_sdk::provider_main::start_provider,
    wasmcloud_provider_sdk::{load_host_data, Context},
};

// workaround to make the following macro compile
wasmtime::component::bindgen!("wamli-mlprovider");

wasmcloud_provider_wit_bindgen::generate!({
    impl_struct: AiModelProvider,
    contract: "wamli:ml",
    wit_bindgen_cfg: "wamli-mlprovider"
});

#[derive(Default, Clone)]
pub struct AiModelProvider {
    // store redis connections per actor
    actors: Arc<RwLock<HashMap<String, String>>>,
    // Default connection URL for actors without a `URL` link value
    default_connect_url: String,
}

impl AiModelProvider {
    pub fn new(default_connect_url: &str) -> Self {
        // let _ = get_data();
        AiModelProvider {
            default_connect_url: default_connect_url.to_string(),
            ..Default::default()
        }
    }
}

/// Handle provider control commands
/// put_link (new actor link command), del_link (remove link command), and shutdown
#[async_trait]
impl WasmcloudCapabilityProvider for AiModelProvider {
    /// Provider should perform any operations needed for a new link,
    /// including setting up per-actor resources, and checking authorization.
    /// If the link is allowed, return true, otherwise return false to deny the link.
    #[instrument(level = "debug", skip(self, ld), fields(actor_id = %ld.actor_id))]
    async fn put_link(&self, ld: &LinkDefinition) -> bool {
        let ih = InvocationHandler::new(ld);
        // let x = ih.get_data().await.unwrap();
        // let x = ih.get_metadata().await.unwrap();
        true
    }

    /// Handle notification that a link is dropped - close the connection
    #[instrument(level = "info", skip(self))]
    async fn delete_link(&self, actor_id: &str) {
        let mut _aw = self.actors.write().await;
    }

    /// Handle shutdown request by closing all connections
    async fn shutdown(&self) {
        let mut _aw = self.actors.write().await;
    }
}

#[async_trait]
impl WamliMlInference for AiModelProvider {
    // async fn fake_it(&self, _ctx: Context) -> bool {
    //     true
    // }

    async fn predict(&self, _ctx: Context, _input: InferenceInput) -> InferenceOutput {
        let tensor = Tensor {
            dimensions: vec![1, 2, 3],
            value_types: vec![],
            bit_flags: 0,
            data: vec![],
        };

        InferenceOutput {
            status: Status::Success(true),
            tensor: tensor,
        }
    }

    // async fn predict(&self, _ctx: Context, _: u32) -> bool {
    //     true
    // }
}