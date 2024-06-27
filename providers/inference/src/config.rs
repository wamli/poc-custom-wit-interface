use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::data_loader::ModelMetadata;
use inference::wamli::ml::types::DataType;

pub type ModelName = String;
pub type ModelId = String;
pub type ModelZoo = HashMap<ModelName, ModelContext>;

/// This is not set explicitly in the build script.
/// It seems to be derviced by the server.
pub const MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+gzip";

/// Default URL to use to connect to registry
pub const DEFAULT_CONNECT_URL: &str = "localhost:5000";

/// Configuration key that will be used to search for config url
pub const CONFIG_URL_KEY: &str = "URL";

/// Graph (model number)
pub type Graph = u32;

/// GraphEncoding
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEncoding {
    #[default]
    Onnx,
    TfLite,
    OpenVino,
    Tensorflow,
}

/// GraphExecutionContext
pub type GraphExecutionContext = u32;

/// ExecutionTarget
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionTarget {
    #[default]
    Cpu,
    Gpu,
    Tpu,
}


/// Configuration for this provider, which is passed to the provider from the host.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderConfig {
    pub values: HashMap<String, String>,
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

// #[derive(Clone, Debug, PartialEq, Deserialize)]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelContext {
    pub model_id: ModelId,
    pub graph_encoding: GraphEncoding,
    pub execution_target: ExecutionTarget,
    pub dtype: DataType,
    pub graph_execution_context: GraphExecutionContext,
    pub graph: Graph,
}

impl ModelContext {
    pub fn default() -> ModelContext {
        ModelContext {
            model_id: Default::default(),
            graph_encoding: Default::default(),
            execution_target: Default::default(),
            dtype: DataType::F32,
            graph_execution_context: Default::default(),
            graph: Default::default(),
        }
    }

    // /// load metadata
    // pub fn load_metadata(&mut self, metadata: ModelMetadata) -> Result<&ModelContext, MlError> {
    //     self.graph_encoding = metadata.graph_encoding;
    //     self.dtype =
    //         ValueType::try_from(metadata.tensor_type.as_str()).map_err(MlError::InvalidModel)?;
    //     self.execution_target = metadata.execution_target;

    //     Ok(self)
    // }
}