use log::error;
use std::io::Read;

#[allow(dead_code)]
use tracing::info;

mod model_loader;
mod oci_image_loader;

pub use crate::data_loader::model_loader::ModelMetadata;

// pub const MEDIA_TYPE: &str = "application/vnd.oci.image.layer.v1.tar+gzip";
pub const MEDIA_TYPE: &str = "application/vnd.docker.image.rootfs.diff.tar.gzip";

pub struct ModelRawData {
    pub model: Vec<u8>,
    pub metadata: model_loader::ModelMetadata,
}

pub async fn fetch_model(registry: &str, image_ref: &str) -> DataLoaderResult<ModelRawData> {
    let oci_image = registry.to_owned() + "/" + &image_ref;

    info!(
        "executing PREFETCH with registry '{}', model '{}' and image '{}'",
        registry, image_ref, &oci_image
    );

    let model_data = pull_model_and_metadata(&oci_image, MEDIA_TYPE).await?;

    info!(
        "PREFETCHED - metadata '{:?}' and model of size '{}'",
        &model_data.metadata,
        model_data.model.len()
    );

    Ok(model_data)
}

pub async fn pull_model_and_metadata(
    image_ref: &str,
    content_type: &str,
) -> DataLoaderResult<ModelRawData> {
    let oci_image = oci_image_loader::pull_image(image_ref, content_type).await?;

    let first_layer = oci_image_loader::read_first_layer(oci_image).await?;

    let uncompressed_layer = oci_image_loader::uncompress_layer(first_layer).await?;

    println!(
        "Uncompressed layer size: {} [bytes]\n",
        uncompressed_layer.len()
    );

    let (model, meta_rawdata) = model_loader::untar_model_and_metadata(uncompressed_layer).await?;

    let metadata = ModelMetadata::from_rawdata(&meta_rawdata).await?;

    Ok(ModelRawData {
        model: model,
        metadata: metadata,
    })
}

/// Data Loader Result
pub type DataLoaderResult<T> = Result<T, DataLoaderError>;

#[derive(Debug, thiserror::Error)]
pub enum DataLoaderError {
    #[error("invalid input {0}")]
    ModelLoaderReadError(String),

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
