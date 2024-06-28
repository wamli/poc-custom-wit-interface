wit_bindgen_wrpc::generate!();

pub use crate::exports::wamli::ml::inference::{Handler, Status};
pub use crate::wamli::ml::types::MlError;

// //! Redis implementation for wasmcloud:keyvalue.
// //!
// //! This implementation is multi-threaded and operations between different actors
// use std::sync::Arc;
// // use std::ops::DerefMut;
// use std::collections::HashMap;
// use async_nats::subject::ToSubject;
// use anyhow::{anyhow, bail, Context as _};

// use tokio::sync::RwLock;
// use tracing::{error, info, instrument, warn};

// // use wasmtime::component::{Resource, ResourceTable};

// use wasmcloud_provider_sdk::core::HostData;
// use wasmcloud_provider_sdk::{
//     get_connection, load_host_data, run_provider, Context, LinkConfig, Provider,
// };

// // use wasmcloud_provider_wit_bindgen::deps::{
// //     async_trait::async_trait,
// //     serde::Deserialize,
// //     serde_json,
// //     wasmcloud_provider_sdk::core::LinkDefinition,
// //     wasmcloud_provider_sdk::provider_main::start_provider,
// //     wasmcloud_provider_sdk::{load_host_data, Context},
// // };

// // workaround to make the following macro compile
// // wasmtime::component::bindgen!("wamli-mlprovider");
// wit_bindgen_wrpc::generate!();

// // wasmcloud_provider_wit_bindgen::generate!({
// //     impl_struct: AiModelProvider,
// //     contract: "wamli:ml",
// //     wit_bindgen_cfg: "wamli-mlprovider"
// // });

// #[derive(Default, Clone)]
// pub struct AiModelProvider {
//     // store redis connections per actor
//     actors: Arc<RwLock<HashMap<String, String>>>,
//     // Default connection URL for actors without a `URL` link value
//     // default_config: ConnectionConfig,
// }

// impl AiModelProvider {
//     /// Execute the provider, loading default configuration from the host and subscribing
//     /// on the proper RPC topics via `wrpc::serve`
//     pub async fn run() -> anyhow::Result<()> {
//         let host_data = load_host_data().context("failed to load host data")?;
//         let provider = Self::from_host_data(host_data);
//         let shutdown = run_provider(provider.clone(), "fake ml provider")
//             .await
//             .context("failed to run provider")?;
//         let connection = get_connection();
//         serve(
//             &connection.get_wrpc_client(connection.provider_key()),
//             provider,
//             shutdown,
//         )
//         .await
//     }

//     /// Build a [`AiModelProvider`] from [`HostData`]
//     pub fn from_host_data(host_data: &HostData) -> AiModelProvider {
//         let default_config = ConnectionConfig::from(&host_data.config);
//         AiModelProvider {
//             default_config,
//             ..Default::default()
//         }
//     }

//     // pub fn new(default_connect_url: &str) -> Self {
//     //     // let _ = get_data();
//     //     AiModelProvider {
//     //         default_connect_url: default_connect_url.to_string(),
//     //         ..Default::default()
//     //     }
//     // }
// }

// // Handle provider control commands
// // put_link (new actor link command), del_link (remove link command), and shutdown
// // #[async_trait]
// // impl Provider for AiModelProvider {
// //     /// Provider should perform any operations needed for a new link,
// //     /// including setting up per-actor resources, and checking authorization.
// //     /// If the link is allowed, return true, otherwise return false to deny the link.
// //     #[instrument(level = "debug", skip(self, ld), fields(actor_id = %ld.actor_id))]
// //     async fn put_link(&self, ld: &LinkDefinition) -> bool {
// //         let ih = InvocationHandler::new(ld);
// //         // let x = ih.get_data().await.unwrap();
// //         // let x = ih.get_metadata().await.unwrap();
// //         true
// //     }

// //     /// Handle notification that a link is dropped - close the connection
// //     #[instrument(level = "info", skip(self))]
// //     async fn delete_link(&self, actor_id: &str) {
// //         let mut _aw = self.actors.write().await;
// //     }

// //     /// Handle shutdown request by closing all connections
// //     async fn shutdown(&self) {
// //         let mut _aw = self.actors.write().await;
// //     }
// // }

// // #[async_trait]
// // impl WamliMlInference for AiModelProvider {
// //     // async fn fake_it(&self, _ctx: Context) -> bool {
// //     //     true
// //     // }

// //     async fn predict(&self, _ctx: Context, _input: String) -> String {
// //         String::from("Greetings from ml provider!")
// //     }

// //     async fn predict(&self, _ctx: Context, _input: InferenceInput) -> InferenceOutput {
        
// //         let host_data = load_host_data().unwrap();
// //         let lattice_rpc_url = &host_data.lattice_rpc_url;
    
// //         log::info!("Host ID: {}", &host_data.host_id);
// //         log::info!("Lattice RPC URL: {}", lattice_rpc_url);
    
// //         // let _connection = wasmcloud_provider_sdk::provider_main::get_connection();
// //         // let client = connection.get_rpc_client().client();
// //         let client = async_nats::connect(lattice_rpc_url).await.unwrap();
// //         // .map_err(|error| {
// //         //     log::error!("get_model_and_metadata() failed!");
// //         //     RpcError::ProviderInit(format!("{}", error))
// //         // })?;
    
// //         let client = wasmcloud_control_interface::Client::new(client);
    
// //         client.scale_actor(
// //             &host_data.host_id,
// //             "localhost:5000/v2/squeezenet_model:0.1.0",
// //             1,
// //             None,
// //         ).await.unwrap();
// //         // .map_err(|error| {
// //         //     log::error!("get_model_and_metadata() failed!");
// //         //     RpcError::ProviderInit(format!("{}", error))
// //         // })?;
        
// //         // client.advertise_link(
// //         //     "MD5EWC4IJDCGIQQGGESG5WHFGUKTKZSEA2A7KNIH7YAKAQJ72MZDUM7G", 
// //         //     "VDEUFI5U4MUVTPCYQMRCFAAZ5SOVGSEEISRFF76YXKD2B4DHQPXY72ZD", 
// //         //     "wamli:mlinference", 
// //         //     "model-link", 
// //         //     HashMap::new(),
// //         // ).await.unwrap();

// //         let tensor = Tensor {
// //             dimensions: vec![1, 2, 3],
// //             value_types: vec![],
// //             bit_flags: 0,
// //             data: vec![],
// //         };

// //         InferenceOutput {
// //             status: Status::Success(true),
// //             tensor: tensor,
// //         }
// //     }

// //     async fn predict(&self, _ctx: Context, _: u32) -> bool {
// //         true
// //     }
// // }