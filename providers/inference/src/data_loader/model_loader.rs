use super::*;

use serde::{Deserialize, Serialize};
use std::io::Cursor;
use tar::Archive;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Model name
    #[serde(default)]
    pub model_name: String,

    /// graph encoding
    #[serde(default)]
    pub graph_encoding: String,

    /// execution target
    #[serde(default)]
    pub execution_target: String,

    /// tensor type
    #[serde(default)]
    pub tensor_dtype: String,

    /// tensor dimensions in (optional)
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
            .map_err(|e| DataLoaderError::ModelLoaderJsonError(format!("invalid json: {}", e)))
    }
}

// pub async fn untar_metadata(
//    tar_archive: &mut Archive<Cursor<Vec<u8>>>
// ) -> DataLoaderResult<Vec<u8>> {
//    let mut tar_entries = tar_archive.entries()
//    .map_err(|error| {
//       log::error!("The tar archive does not contain any entries!");
//       DataLoaderError::ModelLoaderTarError(format!("{}", error))
//    })?;

//    let mut config_file = tar_entries
//    .find(|entry|
//       entry
//       .as_ref()
//       .is_ok_and(|e| e
//          .path()
//          .is_ok_and(|path| path
//             .extension()
//             .is_some_and(|ext| ext.to_str().is_some_and(|e| e == "json"))
//          )
//       )
//    )
//    .ok_or_else(|| format!("No JSON file found in the tar archive"))
//    .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?
//    .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

//    // let mut content = config_file.map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;
//    let mut metadata:Vec<u8> = Vec::new();
//    config_file.read_to_end(&mut metadata).map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

//    Ok(metadata)
// }

// pub async fn untar_model(
//    tar_archive: &mut Archive<Cursor<Vec<u8>>>
// ) -> DataLoaderResult<Vec<u8>> {
//    let mut tar_entries = tar_archive.entries()
//    .map_err(|error| {
//       log::error!("The tar archive does not contain any entries!");
//       DataLoaderError::ModelLoaderTarError(format!("{}", error))
//    })?;

//    let mut model_file = tar_entries
//    .find(|entry|
//       entry
//       .as_ref()
//       .is_ok_and(|e| e
//          .path()
//          .is_ok_and(|path| path
//             .extension()
//             .is_some_and(|ext| ext.to_str().is_some_and(|e| e != "json"))
//          )
//       )
//    )
//    .ok_or_else(|| format!("No JSON file found in the tar archive"))
//    .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?
//    .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

//    // let mut content = config_file.map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;
//    let mut model:Vec<u8> = Vec::new();
//    model_file.read_to_end(&mut model).map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

//    Ok(model)
// }

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
        .ok_or_else(|| format!("No JSON file found in the tar archive"))
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;

    let mut model: Vec<u8> = Vec::new();
    model_file
        .read_to_end(&mut model)
        .map_err(|e| DataLoaderError::ModelLoaderTarError(format!("{}", e)))?;
    Ok((model, metadata))
}
