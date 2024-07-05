use super::*;

use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tar::Archive;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Model name
    /// Optional, `model_id` is used instead.
    /// `model_id` is supposed to be a query parameter
    #[serde(default)]
    pub model_name: Option<String>,

    /// graph encoding
    /// Mandatory
    pub graph_encoding: String,

    /// execution target
    /// Shall default to 'Cpu' in case, it is erroneous or not provided
    #[serde(default)]
    pub execution_target: String,

    /// tensor type
    /// Shall default to 'f32' in case, it is erroneous or not provided
    #[serde(default)]
    pub tensor_dtype: String,

    /// tensor dimensions in (optional)
    /// Each model expects a specific shape of input tensors.
    /// However, the shape is currently not processed any further.
    #[serde(default)]
    pub tensor_shape_in: Option<Vec<u32>>,

    /// tensor dimensions out (optional)
    #[serde(default)]
    pub tensor_shape_out: Option<Vec<u32>>,
}

impl ModelMetadata {
    /// load metadata from json
    pub async fn from_rawdata(data: &[u8]) -> Result<Self, DataLoaderError> {
        serde_json::from_slice(data)
            .map_err(|e| DataLoaderError::ModelLoaderJsonError(format!("invalid json (metadata): {}", e)))
    }
}

/// get model and metadata
pub async fn untar_model_and_metadata(data: Vec<u8>) -> DataLoaderResult<(Vec<u8>, Vec<u8>)> {
    let mut tar_archive = Archive::new(Cursor::new(data));

    let mut tar_entries = tar_archive.entries().map_err(|error| {
        log::error!("The tar archive does not contain any entries!");
        DataLoaderError::ModelLoaderTarError(format!("{}", error))
    })?;

    let mut config_file = tar_entries
        .find(|entry| {
            entry.as_ref().is_ok_and(|e| {
                e.path().is_ok_and(|path| {
                    path.extension()
                        .is_some_and(|ext| ext.to_str().is_some_and(|e| e == "json"))
                })
            })
        })
        .ok_or_else(|| format!("No JSON file found in the tar archive"))
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

    let mut metadata: Vec<u8> = Vec::new();
    config_file
        .read_to_end(&mut metadata)
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

    let mut model_file = tar_entries
        .find(|entry| {
            entry.as_ref().is_ok_and(|e| {
                e.path().is_ok_and(|path| {
                    path.extension()
                        .is_some_and(|ext| ext.to_str().is_some_and(|e| e != "json"))
                })
            })
        })
        .ok_or_else(|| format!("No model found in the tar archive"))
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

    let mut model: Vec<u8> = Vec::new();
    model_file
        .read_to_end(&mut model)
        .map_err(|e| DataLoaderError::ModelLoaderReadError(format!("{}", e)))?;
    Ok((model, metadata))
}
