use crate::config::{ProviderConfig, CONFIG_URL_KEY, DEFAULT_CONNECT_URL};
use crate::data_loader::{self, ModelRawData};
use crate::engine::{
    get_engine, get_or_else_set_engine, Engine, ExecutionTarget, Graph, GraphEncoding,
    GraphExecutionContext, InferenceFramework, ModelContext, ModelZoo,
};
use crate::{serve, DataType, Handler, MlError, Tensor};
use anyhow::anyhow;
use anyhow::Context as _;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use wasmcloud_provider_sdk::{run_provider, Context, LinkConfig, Provider, ProviderInitConfig};

pub struct ModelData {
    pub model: Vec<u8>,
    pub metadata: ModelContext,
}

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

    /// There are the following relevant types:
    ///     - InferenceFramework
    ///     - Engine
    ///     - InferenceEngine
    ///     - GraphEncoding
    ///
    /// InferenceFramework corresponds to an integrated crate.
    /// An InferenceFramework may support a multitude of Engine
    /// An InferenceFramework may support a multitude of GraphEncoding.
    /// Engine is a wrapper of InferenceEngine
    /// InferenceEngine defines common behavior
    /// GraphEncoding defines a model's encoding.
    engines: Arc<RwLock<HashMap<InferenceFramework, Engine>>>,

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

    pub async fn get_registry(&self) -> String {
        let config_guard = self.config.read().await;

        let registry = match config_guard.values.get(CONFIG_URL_KEY) {
            Some(url) => url.to_string(),
            None => DEFAULT_CONNECT_URL.to_string(),
        };

        registry
    }

    pub async fn register_model(
        &self,
        model_id: &str,
        model_data: ModelRawData,
    ) -> anyhow::Result<()> {
        let metadata = model_data.metadata;

        let graph_encoding = GraphEncoding::from_str(&metadata.graph_encoding)
            .map_err(|error| anyhow!(error.to_string()))?;

        let execution_target = ExecutionTarget::from_str(&metadata.execution_target)
            .map_err(|error| anyhow!(error.to_string()))?;

        let data_type = DataType::from_str(&metadata.execution_target)
            .map_err(|error| anyhow!(error.to_string()))?;

        let engine = get_or_else_set_engine(Arc::clone(&self.engines), &graph_encoding).await?;

        let graph: Graph = engine
            .load(&model_data.model)
            .await
            .map_err(|error| anyhow!(error.to_string()))?;

        let gec: GraphExecutionContext = engine
            .init_execution_context(graph, &execution_target, &graph_encoding)
            .await
            .map_err(|error| anyhow!(error.to_string()))?;

        let model_context = ModelContext {
            model_name: model_id.to_owned(),
            graph_encoding: graph_encoding,
            execution_target: execution_target,
            dtype: data_type,
            graph: graph,
            graph_execution_context: gec,
        };

        let mut models_lock = self.models.write().await;

        if let Some(already_context) = models_lock.insert(model_id.to_owned(), model_context) {
            log::warn!(
                "model '{}' is already registered: {:?}",
                model_id,
                already_context
            );
        };
        Ok(())
    }
}

/// When a provider specifies an `export` in its `wit/world.wit` file, the `wit-bindgen-wrpc` tool generates
/// a trait that the provider must implement. This trait is used to handle invocations from components that
/// link to the provider. The `Handler` trait is generated for each export in the WIT world.
impl Handler<Option<Context>> for InferenceProvider {
    async fn predict(
        &self,
        _ctx: Option<Context>,
        model_id: String,
        tensor_in: Tensor,
    ) -> anyhow::Result<Result<Tensor, MlError>> {
        info!("PREDICTING ... the future");

        let models_lock = self.models.read().await;

        let model_context = match models_lock.get(&model_id) {
            Some(mc) => mc.to_owned(),
            None => {
                log::error!(
                    "predict() - model {} not found in models={:?}",
                    &model_id,
                    &models_lock
                );
                return Ok(Err(MlError::ContextNotFoundError(format!(
                    "Model '{}' is unknown",
                    &model_id
                ))));
            }
        };

        let engine = get_engine(Arc::clone(&self.engines), &model_context.graph_encoding).await?;

        let inference_result = tokio::task::spawn_blocking(move || async move {
            if let Err(e) = engine
                .set_input(model_context.graph_execution_context, 0, &tensor_in)
                .await
            {
                log::error!(
                    "predict() - inference engine failed in 'set_input()' with '{}'",
                    e
                );
                return Err(MlError::ContextNotFoundError(e.to_string()));
            }

            if let Err(e) = engine.compute(model_context.graph_execution_context).await {
                log::error!("predict() - GraphExecutionContext not found: {}", e);
                return Err(MlError::ContextNotFoundError(e.to_string()));
            }

            let result = engine
                .get_output(model_context.graph_execution_context, 0)
                .await;

            Ok(result)

        })
        .await
        .map_err(|e| MlError::Internal(format!("internal join error: {}", e)))?
        .await?;

        if let Err(error) = inference_result {
            log::error!("predict() - problem collecting results {}", error);
            return Ok(Err(MlError::Internal(error.to_string())));
        };

        match inference_result {
            Err(error) => {
                log::error!("predict() - problem collecting results {}", error);
                Ok(Err(MlError::Internal(error.to_string())))
            }
            Ok(tensor_out) => Ok(Ok(tensor_out)),
        }
    }

    async fn prefetch(
        &self,
        _ctx: Option<Context>,
        model_id: String,
    ) -> anyhow::Result<Result<(), MlError>> {
        info!("prefetching model '{}'", model_id);

        let registry = self.get_registry().await;

        let model_data = data_loader::fetch_model(&registry, &model_id)
            .await
            .map_err(|error| MlError::Internal(error.to_string()))?;

        self.register_model(&model_id, model_data).await?;

        Ok(Ok(()))
    }
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

        let registry = self.get_registry().await;

        for (_, image_ref) in config_lock
            .values
            .iter()
            .filter(|(k, _)| !k.starts_with("url"))
        {
            let model_data = data_loader::fetch_model(&registry, &image_ref)
                .await
                .map_err(|error| MlError::Internal(error.to_string()))?;

            self.register_model(&image_ref, model_data).await?;
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

impl InferenceProvider {}
