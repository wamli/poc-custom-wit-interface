#[cfg(feature = "tflite")]
mod tflite;

mod tract;

#[cfg(any(feature = "tflite", feature = "edgetpu"))]
pub use self::tflite::TfLiteEngine;

use crate::data_loader::DataLoaderError;
use crate::engine::tract::TractEngine;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(any(feature = "tflite", feature = "edgetpu"))]
use crate::engine::tflite::TfLiteEngine;

// pub use tract::{bytes_to_f32_vec, f32_array_to_bytes, TractEngine, TractSession};

// use wasmcloud_interface_mlinference::{InferenceOutput, Tensor};
use crate::{DataType, Tensor};

/// Graph (model number)
pub type Graph = u32;

pub type Engine = Arc<Box<dyn InferenceEngine + Send + Sync>>;

/// GraphExecutionContext
pub type GraphExecutionContext = u32;

pub type ModelId = String;
pub type ModelZoo = HashMap<ModelId, ModelContext>;

impl FromStr for DataType {
    type Err = DataLoaderError;

    fn from_str(dt: &str) -> Result<Self, Self::Err> {
        match dt.to_lowercase().as_str() {
            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "u128" => Ok(DataType::U128),
            "s8" => Ok(DataType::S8),
            "s16" => Ok(DataType::S16),
            "s32" => Ok(DataType::S32),
            "s64" => Ok(DataType::S64),
            "s128" => Ok(DataType::S128),
            "f16" => Ok(DataType::F16),
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),
            "f128" => Ok(DataType::F128),
            "na" => Ok(DataType::Na),

            _ => {
                log::warn!(
                    "invalid or missing data type detected: '{}' - defaults to 'f32'",
                    dt,
                );

                Ok(DataType::F32)
            },
        }
    }
}

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

impl FromStr for GraphEncoding {
    type Err = DataLoaderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "onnx" => Ok(GraphEncoding::Onnx),
            "tflite" => Ok(GraphEncoding::TfLite),
            "openvino" => Ok(GraphEncoding::OpenVino),
            "tensorflow" => Ok(GraphEncoding::Tensorflow),
            _ => Err(DataLoaderError::ModelLoaderMetadataError(format!(
                "Invalid graph encoding: '{}'",
                s
            ))),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Hash)]
pub enum InferenceFramework {
    #[default]
    Tract,
    TfLite,
}

/// ExecutionTarget
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionTarget {
    #[default]
    Cpu,
    Gpu,
    Npu,
    Tpu,
}

impl FromStr for ExecutionTarget {
    type Err = DataLoaderError;

    fn from_str(et: &str) -> Result<Self, Self::Err> {
        match et.to_lowercase().as_str() {
            "cpu" => Ok(ExecutionTarget::Cpu),
            "tpu" => Ok(ExecutionTarget::Tpu),
            "gpu" => Ok(ExecutionTarget::Gpu),
            "npu" => Ok(ExecutionTarget::Npu),
            _ => {
                log::warn!(
                    "invalid or missing execution target detected: '{}' - defaults to cpu",
                    et,
                );

                Ok(ExecutionTarget::Cpu)
            },
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

    // /// load metadata
    // pub fn load_metadata(
    //     &mut self,
    //     metadata: ModelMetadata,
    // ) -> Result<&ModelContext, DataLoaderError> {
    //     self.model_name = metadata.model_name;
    //     self.graph_encoding = GraphEncoding::Onnx;
    //     self.execution_target = ExecutionTarget::Cpu;

    //     Ok(self)
    // }
}

/// InferenceEngine
#[async_trait]
pub trait InferenceEngine {
    async fn load(&self, model: &[u8]) -> InferenceResult<Graph>;

    async fn init_execution_context(
        &self,
        graph: Graph,
        target: &ExecutionTarget,
        encoding: &GraphEncoding,
    ) -> InferenceResult<GraphExecutionContext>;

    async fn set_input(
        &self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor,
    ) -> InferenceResult<()>;

    async fn compute(&self, context: GraphExecutionContext) -> InferenceResult<()>;

    async fn get_output(
        &self,
        context: GraphExecutionContext,
        index: u32,
    ) -> InferenceResult<Tensor>;

    async fn drop_model_state(&self, graph: &Graph, gec: &GraphExecutionContext);
}

// impl Default for Box<dyn InferenceEngine + Send + Sync> {
//     fn default() -> Box<dyn InferenceEngine + Send + Sync>
//     where
//         Self: Sized,
//     {
//         Box::new(<TractEngine as Default>::default())
//     }
// }

pub async fn get_engine(
    engines: Arc<RwLock<HashMap<InferenceFramework, Engine>>>,
    encoding: &GraphEncoding,
) -> InferenceResult<Engine> {
    let engines_lock = engines.write().await;

    log::debug!("get_engine() - context: {:?}", &encoding);

    match encoding {
        GraphEncoding::Onnx | GraphEncoding::Tensorflow => {
            match engines_lock.get(&InferenceFramework::Tract) {
                Some(e) => Ok(e.clone()),
                None => Err(InferenceError::InvalidEncodingError),
            }
        }

        #[cfg(any(feature = "tflite", feature = "edgetpu"))]
        GraphEncoding::TfLite => match engines_lock.get(&InferenceFramework::TfLite) {
            Some(e) => Ok(e.clone()),
            None => Err(InferenceError::InvalidEncodingError),
        },

        _ => {
            log::error!(
                "get_engine() - unsupported graph encoding detected '{:?}'",
                &encoding
            );
            Err(InferenceError::InvalidEncodingError)
        }
    }
}

/// Each link definition may address a different target
/// such that it may be necessary to support multiple engines.
pub async fn get_or_else_set_engine(
    engines: Arc<RwLock<HashMap<InferenceFramework, Engine>>>,
    encoding: &GraphEncoding,
) -> InferenceResult<Engine> {
    let mut engines_lock = engines.write().await;

    match encoding {
        GraphEncoding::Onnx | GraphEncoding::Tensorflow => {
            match engines_lock.get(&InferenceFramework::Tract) {
                Some(e) => {
                    log::debug!("get_or_else_set_engine() - previously created Onnx/Tensorflow engine selected for '{:?}'.", &encoding);
                    Ok(e.clone())
                }
                None => {
                    let feedback = engines_lock.insert(
                        InferenceFramework::Tract,
                        Arc::new(Box::new(TractEngine::default())),
                    );
                    log::debug!("get_or_else_set_engine() - Onnx/Tensorflow engine selected and created for '{:?}'.", &encoding);

                    if feedback.is_none() {
                        log::debug!(
                            "get_or_else_set_engine() - Onnx/Tensorflow was definitely created"
                        );
                    } else {
                        log::debug!("get_or_else_set_engine() - Onnx/Tensorflow was NOT created!");
                    }

                    log::debug!(
                        "get_or_else_set_engine() - content of Engines: {:?}: with length {:?}",
                        &engines_lock.keys(),
                        &engines_lock.keys().len()
                    );

                    //assert_eq!(engines_lock.is_empty(), false);

                    Ok(engines_lock
                        .get(&InferenceFramework::Tract)
                        .unwrap()
                        .clone())
                }
            }
        }

        #[cfg(any(feature = "tflite", feature = "edgetpu"))]
        GraphEncoding::TfLite => match engines_lock.get(&InferenceFramework::TfLite) {
            Some(e) => {
                log::debug!("get_or_else_set_engine() - previously created TfLite engine selected for '{}'.", encoding.model_name);
                Ok(e.clone())
            }
            None => {
                let feedback = engines_lock.insert(
                    InferenceFramework::TfLite,
                    Arc::new(Box::new(TfLiteEngine::default())),
                );
                log::debug!(
                    "get_or_else_set_engine() - TfLite engine selected and created for '{}'.",
                    encoding.model_name
                );

                if feedback.is_none() {
                    log::debug!("get_or_else_set_engine() - TfLite engine was definitely created");
                } else {
                    log::debug!("get_or_else_set_engine() - TfLite was NOT created!");
                }

                log::debug!(
                    "get_or_else_set_engine() - content of Engines: {:?}: with length {:?}",
                    &engines_lock.keys(),
                    &engines_lock.keys().len()
                );

                Ok(engines_lock
                    .get(&InferenceFramework::TfLite)
                    .unwrap()
                    .clone())
            }
        },
        _ => {
            log::error!("unsupported graph encoding detected '{:?}'", &encoding);
            Err(InferenceError::InvalidEncodingError)
        }
    }
}

/// InferenceResult
pub type InferenceResult<T> = Result<T, InferenceError>;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("runtime error")]
    RuntimeError,

    #[error("ONNX error")]
    OnnxError,

    #[error("Unsupported ExecutionTarget")]
    UnsupportedExecutionTarget,

    #[error("Invalid encoding")]
    InvalidEncodingError,

    #[error("Failed to build model from buffer")]
    FailedToBuildModelFromBuffer,

    #[error("Failed to get edge TPU context")]
    EdgeTPUAllocationError,

    #[error("Failed to get InterpreterBuilder")]
    InterpreterBuilderError,

    #[error("Interpreter build failed")]
    InterpreterBuildError,

    #[error("Interpreter invocation failed")]
    InterpreterInvocationError,

    #[error("Tensor allocation failed")]
    TensorAllocationError,

    #[error("Corrupt input tensor")]
    CorruptInputTensor,

    #[error("Re-shaping of tensor failed {0}")]
    ReShapeError(String),

    #[error("Bytes to f32 vec conversion failed")]
    BytesToVecConversionError(#[from] std::io::Error),

    #[error("Configuration of model's input type and/or shape failed")]
    CorruptInputTypeOrShape(#[from] tract_onnx::tract_core::anyhow::Error),
}
