// wit_bindgen_wrpc::generate!();

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Context as _;
use tracing::{debug, info, warn, error};
use std::collections::HashMap;
use crate::config::{ModelContext, ModelZoo, ProviderConfig, CONFIG_URL_KEY, DEFAULT_CONNECT_URL, MEDIA_TYPE};
use crate::data_loader::{pull_model_and_metadata, DataLoaderError, DataLoaderResult};

use wasmcloud_provider_sdk::{run_provider, Context, LinkConfig, Provider, ProviderInitConfig};

use inference::{Handler, serve, Status, MlError};

#[derive(Default, Clone)]
pub struct InferenceProvider {
    /// map to store the assignments between the respective model
    /// and corresponding bindle path for each linked actor
    /// TODO:
    ///   - instead of delaying putLink for model loading and initialization,
    ///     add a Ready flag (AtomicBool) that is set after model is loaded and initialized.
    ///   - initialize actor link as soon as we receive the putlink command
    ///   - if health check or rpc is received when not ready, return not-ready error
    // components: Arc<RwLock<HashMap<String, ModelZoo>>>,
    models: Arc<RwLock<ModelZoo>>,

    config: Arc<RwLock<ProviderConfig>>,
    /// All components linked to this provider and their config.
    linked_from: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    /// All components this provider is linked to and their config
    linked_to: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
}

impl InferenceProvider {
   /// Execute the provider, loading [`HostData`] from the host which includes the provider's configuration and
   /// information about the host. Once you use the passed configuration to construct a [`InferenceProvider`],
   /// you can run the provider by calling `run_provider` and then serving the provider's exports on the proper
   /// RPC topics via `wrpc::serve`.
   ///
   /// This step is essentially the same for every provider, and you shouldn't need to modify this function.
    pub async fn run() -> anyhow::Result<()> {
       let provider = Self::default();
       let shutdown = run_provider(provider.clone(), "custom-template-provider")
           .await
           .context("failed to run provider")?;

       // The [`serve`] function will set up RPC topics for your provider's exports and await invocations.
       // This is a generated function based on the contents in your `wit/world.wit` file.
       let connection = wasmcloud_provider_sdk::get_connection();
       serve(
           &connection.get_wrpc_client(connection.provider_key()),
           provider,
           shutdown,
       )
       .await

       // If your provider has no exports, simply await the shutdown to keep the provider running
       // shutdown.await;
       // Ok(())
   }

   pub async fn fetch_model(&self, image_ref: String) -> DataLoaderResult<()> {
        let config_guard = self.config.read().await;

        let registry = match config_guard.values.get(CONFIG_URL_KEY) {
            Some(url) => url,
            None => DEFAULT_CONNECT_URL,
        };

        let oci_image = registry.to_owned() + "/" + &image_ref;

        info!("executing PREFETCH with registry '{}', model '{}' and image '{}'", registry, image_ref, &oci_image);

        let model_data = pull_model_and_metadata(&oci_image, MEDIA_TYPE).await.unwrap();

        info!("PREFETCHED - metadata '{:?}' and model of size '{}'", model_data.metadata, model_data.model.len());

        let mut models_lock = self.models.write().await;

        let is_model_already_loaded = models_lock.get(&oci_image);

        // Evaluate if the model defined by `oci_image` is already known.
        // If not, add it to the list of modules
        if is_model_already_loaded.is_some() {
            warn!("PREFETCHED - model '{}' is already loaded", &oci_image);
        }

        if is_model_already_loaded.is_none() {
            let mut default_context = ModelContext::default();
            let model_context = default_context
                .load_metadata(model_data.metadata);

            match model_context {
                Ok(mc) => {
                    models_lock.insert(oci_image.to_owned(), mc.clone());
                },

                Err(error) => {
                    return Err(DataLoaderError::ModelLoaderMetadataError(format!("Error loading model: {}", error)));
                }
            };
            info!("PREFETCHED - no. of registered models '{}'", models_lock.len());

            return Ok(());
        }
        Ok(())
    }
}

/// When a provider specifies an `export` in its `wit/world.wit` file, the `wit-bindgen-wrpc` tool generates
/// a trait that the provider must implement. This trait is used to handle invocations from components that
/// link to the provider. The `Handler` trait is generated for each export in the WIT world.
impl Handler<Option<Context>> for InferenceProvider {

    async fn predict(&self, _ctx: Option<Context>, _input: String) -> anyhow::Result<String> {
        let x = "anything";
        let y = "something";
        warn!("-----------------> inference provider PREDICTING ... the future");
        info!(x, y, "-----------------> inference provider PREDICTING ... the future");
        Ok(String::from("Greetings from ml provider!"))
   }

   async fn prefetch(&self, _ctx: Option<Context>, image_ref: String) -> anyhow::Result<Status> {
    info!("prefetching model '{}'", image_ref);

    if let Err(error) = self.fetch_model(image_ref).await {
        return Ok(Status::Error(MlError::InvalidMetadata(format!("Loading model's metadata from OCI image failed: {}", error))));
    };

    // let config_guard = self.config.read().await;

    // let registry = match config_guard.values.get(CONFIG_URL_KEY) {
    //     Some(url) => url,
    //     None => DEFAULT_CONNECT_URL,
    // };

    // let oci_image = registry.to_owned() + "/" + &image_ref;

    // info!("executing PREFETCH with registry '{}', model '{}' and image '{}'", registry, image_ref, &oci_image);

    // let model_data = pull_model_and_metadata(&oci_image, MEDIA_TYPE).await.unwrap();

    // info!("PREFETCHED - metadata '{:?}' and model of size '{}'", model_data.metadata, model_data.model.len());

    // let mut models_lock = self.models.write().await;

    // let is_model_already_loaded = models_lock.get(&oci_image);

    // // Evaluate if the model defined by `oci_image` is already known.
    // // If not, add it to the list of modules
    // if is_model_already_loaded.is_some() {
    //     warn!("PREFETCHED - model '{}' is already loaded", &oci_image);
    // }

    // if is_model_already_loaded.is_none() {
    //     let mut default_context = ModelContext::default();
    //     let model_context = default_context
    //         .load_metadata(model_data.metadata);

    //     match model_context {
    //         Ok(mc) => models_lock.insert(oci_image.to_owned(), mc.clone()),

    //         Err(error) => {
    //             return Ok(Status::Error(MlError::InvalidMetadata(format!("Loading model's metadata from OCI image failed: {}", error))));
    //         }
    //     };
    // }

    let status = Status::Success(true);

    Ok(status)
}

   ///// Request information about the system the provider is running on
   // async fn request_info(&self, ctx: Option<Context>, kind: Kind) -> anyhow::Result<String> {
   //     // The `ctx` contains information about the component that invoked the request. You can use
   //     // this information to look up the configuration of the component that invoked the request.
   //     let requesting_component = ctx
   //         .and_then(|c| c.component)
   //         .unwrap_or_else(|| "UNKNOWN".to_string());
   //     let component_config = self
   //         .linked_from
   //         .read()
   //         .await
   //         .get(&requesting_component)
   //         .cloned()
   //         .unwrap_or_default();

   //     info!(
   //         requesting_component,
   //         ?kind,
   //         ?component_config,
   //         "received request for system information"
   //     );

   //     let info = match kind {
   //         Kind::Os => std::env::consts::OS,
   //         Kind::Arch => std::env::consts::ARCH,
   //     };
   //     Ok(format!("{info}"))
   // }

   // /// Request the provider to send some data to all linked components
   // ///
   // /// This function is easy to invoke with `wash call`,
   // async fn call(&self, _ctx: Option<Context>) -> anyhow::Result<String> {
   //     info!("received call to send data to linked components");
   //     let mut last_response = None;
   //     for (component_id, config) in self.linked_to.read().await.iter() {
   //         debug!(component_id, ?config, "sending data to component");
   //         let sample_data = Data {
   //             name: "sup".to_string(),
   //             count: 3,
   //         };
   //         let client = wasmcloud_provider_sdk::get_connection().get_wrpc_client(component_id);
   //         match process_data::process(&client, &sample_data).await {
   //             Ok(response) => {
   //                 last_response = Some(response);
   //                 info!(
   //                     component_id,
   //                     ?config,
   //                     ?last_response,
   //                     "successfully sent data to component"
   //                 );
   //             }
   //             Err(e) => {
   //                 error!(
   //                     component_id,
   //                     ?config,
   //                     ?e,
   //                     "failed to send data to component"
   //                 );
   //             }
   //         }
   //     }

   //     Ok(last_response.unwrap_or_else(|| "No components responded to request".to_string()))
   // }
}

impl Provider for InferenceProvider {
        /// Initialize your provider with the given configuration. This is a good place to set up any state or
        /// resources your provider needs to run.
        async fn init(&self, config: impl ProviderInitConfig) -> anyhow::Result<()> {
        let provider_id = config.get_provider_id();
        let initial_config = config.get_config();

        info!(provider_id, ?initial_config, "initializing provider");

        // Save configuration to provider state
        *self.config.write().await = ProviderConfig::from(initial_config);

        let config_lock = self.config.read().await;

        for (_, image_ref) in config_lock.values.iter().filter(|(k, _)| !k.starts_with("url")) {
            let image_ref = image_ref.to_owned();
            if let Err(error) = self.fetch_model(image_ref.to_owned()).await {
                warn!("Unable to load model '{}': {}", &image_ref, error);
            }
        }

        Ok(())
    }

    /// When your provider is linked to a component, this method will be called with the [`LinkConfig`] that
    /// is passed in as source configuration. You can store this configuration in your provider's state to
    /// keep track of the components your provider is linked to.
    ///
    /// A concrete use case for this can be seen in our HTTP server provider, where we are given configuration
    /// for a port or an address to listen on, and we can use that configuration to start a webserver and forward
    /// any incoming requests to the linked component.
    async fn receive_link_config_as_source(
        &self,
        LinkConfig {
            target_id, config, ..
        }: LinkConfig<'_>,
    ) -> anyhow::Result<()> {
        // We're storing the configuration as an example of how to keep track of linked components, but
        // the provider SDK does not require you to store this information.
        self.linked_to
            .write()
            .await
            .insert(target_id.to_string(), config.to_owned());

        info!(
            "finished processing link from provider to component [{}] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            target_id
        );

        debug!(
            "finished processing link from provider to component [{}]",
            target_id
        );
        Ok(())
    }

    /// When a component links to your provider, this method will be called with the [`LinkConfig`] that
    /// is passed in as target configuration. You can store this configuration in your provider's state to
    /// keep track of the components linked to your provider.
    ///
    /// A concrete use case for this can be seen in our key-value Redis provider, where we are given configuration
    /// for a Redis connection, and we can use that configuration to store and retrieve data from Redis. When an
    /// invocation is received from a component, we can look up the configuration for that component and use it
    /// to interact with the correct Redis instance.
    async fn receive_link_config_as_target(
        &self,
        LinkConfig {
            source_id, config, ..
        }: LinkConfig<'_>,
    ) -> anyhow::Result<()> {
        self.linked_from
            .write()
            .await
            .insert(source_id.to_string(), config.to_owned());

        info!(
            "finished processing link from component [{}] to provider",
            source_id
        );
        Ok(())
    }

    /// When a link is deleted from your provider to a component, this method will be called with the target ID
    /// of the component that was unlinked. You can use this method to clean up any state or resources that were
    /// associated with the linked component.
    async fn delete_link_as_source(&self, target: &str) -> anyhow::Result<()> {
        self.linked_to.write().await.remove(target);

        info!(
            "finished processing delete link from provider to component [{}]",
            target
        );
        Ok(())
    }

    /// When a link is deleted from a component to your provider, this method will be called with the source ID
    /// of the component that was unlinked. You can use this method to clean up any state or resources that were
    /// associated with the linked component.
    async fn delete_link_as_target(&self, source_id: &str) -> anyhow::Result<()> {
        self.linked_from.write().await.remove(source_id);

        info!(
            "finished processing delete link from component [{}] to provider",
            source_id
        );
        Ok(())
    }

    /// Handle shutdown request by cleaning out all linked components. This is a good place to clean up any
    /// resources or connections your provider has established.
    async fn shutdown(&self) -> anyhow::Result<()> {
        self.linked_from.write().await.clear();
        self.linked_to.write().await.clear();

        Ok(())
    }
}