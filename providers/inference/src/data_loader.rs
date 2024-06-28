use std::{io::Read};
use log::error;

mod model_loader;
mod oci_image_loader;

pub use crate::data_loader::model_loader::ModelMetadata;

pub struct ModelData {
    pub model: Vec<u8>,
    pub metadata: model_loader::ModelMetadata,
 }

pub async fn pull_model_and_metadata(
    image_ref: &str,
    content_type: &str,
) -> DataLoaderResult<ModelData> {
    let oci_image = oci_image_loader::pull_image(image_ref, content_type).await?;

    let first_layer = oci_image_loader::read_first_layer(oci_image).await?;

    let uncompressed_layer = oci_image_loader::uncompress_layer(first_layer).await?;

    println!("Uncompressed layer size: {} [bytes]\n", uncompressed_layer.len());

    let (model, meta_rawdata) = model_loader::untar_model_and_metadata(uncompressed_layer).await?;

    let metadata = ModelMetadata::from_rawdata(&meta_rawdata).await?;

    Ok(
        ModelData { 
            model: model, 
            metadata: metadata
        }
    )
}

/// Data Loader Result
pub type DataLoaderResult<T> = Result<T, DataLoaderError>;

#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
   #[error("invalid tar archive {0}")]
   ModelLoaderTarError(String),

   #[error("invalid json {0}")]
   ModelLoaderJsonError(String),

   #[error("Error parsing metadata {0}")]
   ModelLoaderMetadataError(String),

    #[error("Unable to pull image: {0}")]
    OciImageLoadError(String),

    #[error("Unable to pull image: {0}")]
    OciUncompressError(String),

    #[error("Unable to load image's layer!")]
    OciLayerLoadError,

}