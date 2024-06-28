use std::str::FromStr;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use inference::wamli::ml::types::DataType;
use crate::data_loader::{ModelMetadata, DataLoaderError};

pub type ModelId = String;
pub type ModelZoo = HashMap<ModelId, ModelContext>;

/// This is not set explicitly in the build script.
/// It seems to be derviced by the server.
// pub const MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+gzip";
pub const MEDIA_TYPE: &str = "application/vnd.docker.image.rootfs.diff.tar.gzip";

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
    Npu,
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
    pub model_name: String,
    pub graph_encoding: GraphEncoding,
    pub execution_target: ExecutionTarget,
    pub dtype: DataType,
    pub graph_execution_context: GraphExecutionContext,
    pub graph: Graph,
}

impl ModelContext {
    pub fn default() -> ModelContext {
        ModelContext {
            model_name: Default::default(),
            graph_encoding: Default::default(),
            execution_target: Default::default(),
            dtype: DataType::F32,
            graph_execution_context: Default::default(),
            graph: Default::default(),
        }
    }

    /// load metadata
    pub fn load_metadata(&mut self, metadata: ModelMetadata) -> Result<&ModelContext, DataLoaderError> {
        self.model_name = metadata.model_name;
        self.graph_encoding = GraphEncoding::Onnx;
        self.execution_target = ExecutionTarget::Cpu;

        Ok(self)
    }
}

impl FromStr for GraphEncoding {
    type Err = DataLoaderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "onnx" => Ok(GraphEncoding::Onnx),
            "tflite" => Ok(GraphEncoding::TfLite),
            "openvino" => Ok(GraphEncoding::OpenVino),
            "tensorflow" => Ok(GraphEncoding::Tensorflow),
            _ => Err(DataLoaderError::ModelLoaderMetadataError(format!("Invalid graph encoding: '{}'", s))),
        }
    }
}

impl FromStr for ExecutionTarget {
    type Err = DataLoaderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(ExecutionTarget::Cpu),
            "tpu" => Ok(ExecutionTarget::Tpu),
            "gpu" => Ok(ExecutionTarget::Gpu),
            "npu" => Ok(ExecutionTarget::Npu),
            _ => Err(DataLoaderError::ModelLoaderMetadataError(format!("Invalid execution target: '{}'", s))),
        }
    }
}