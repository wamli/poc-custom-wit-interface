use super::*;

use flate2::read::GzDecoder;
use oci_distribution::{
    client::{ClientConfig, ImageData},
    secrets::RegistryAuth,
    Client, Reference,
};

pub async fn pull_image(image_ref: &str, content_type: &str) -> DataLoaderResult<ImageData> {
    // Experimenting with an ordinary local docker registry,
    // the protocol is `Http`
    let config = ClientConfig {
        protocol: oci_distribution::client::ClientProtocol::Http,
        ..Default::default()
    };

    let client = Client::new(config);

    let reference: Reference = image_ref.parse().unwrap();

    let image_data = client
        .pull(&reference, &RegistryAuth::Anonymous, vec![content_type])
        .await
        .map_err(|error| DataLoaderError::OciImageLoadError(format!("{}", error)))?;

    log::info!("Image successfully pulled!");

    Ok(image_data)
}

pub async fn uncompress_layer(data: Vec<u8>) -> DataLoaderResult<Vec<u8>> {
    let mut decompressed_data = Vec::new();
    let mut gz_decoder = GzDecoder::new(&data[..]);

    gz_decoder
        .read_to_end(&mut decompressed_data)
        .map_err(|error| DataLoaderError::OciUncompressError(format!("{}", error)))?;

    log::info!("Layer succesfully uncompressed!");

    Ok(decompressed_data)
}

pub async fn read_first_layer(image_data: ImageData) -> DataLoaderResult<Vec<u8>> {
    println!("This image has {} layer(s)", &image_data.layers.len());

    // The example image is supposed to have one layer.
    // The first, or only, layer is supposed to contain
    // a text file, e.g. content.txt
    let first_layer = image_data
        .layers
        .into_iter()
        .nth(0)
        .ok_or(DataLoaderError::OciLayerLoadError)?;

    log::info!("First layer succesfully read!");

    Ok(first_layer.data)
}
