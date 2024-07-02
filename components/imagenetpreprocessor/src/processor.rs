//! This actor is designed to support the following two models:
//! * https://github.com/onnx/models/tree/main/vision/classification/mobilenet

use crate::wasi::logging::logging::*;
use byteorder::{LittleEndian, WriteBytesExt};
use image::{load_from_memory, ImageResult, Pixel};
use ndarray::s;

// pub async fn f32_array_to_bytes(values: &[f32]) -> Vec<u8> {
pub fn f32_array_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut wtr = Vec::with_capacity(values.len() * 4);
    for val in values.iter() {
        // unwrap ok because buf is pre-allocated and won't error
        wtr.write_f32::<LittleEndian>(*val).unwrap();
    }
    wtr
}

// pub async fn preprocess(raw_data: &[u8], channels: u32, height: u32, width: u32) -> ImageResult<Vec<u8>> {
pub fn preprocess(raw_data: &[u8], channels: u32, height: u32, width: u32) -> ImageResult<Vec<u8>> {
    log(
        Level::Debug,
        "ImagenetPreProcessor",
        "preprocess() - entry point",
    );

    let raw_image = load_from_memory(raw_data)?;
    // .map_err(|e| RpcError::Deser(e.to_string()))?;

    let image = image::imageops::resize(
        &raw_image,
        width,
        height,
        ::image::imageops::FilterType::Triangle,
    );

    log(
        Level::Debug,
        "ImagenetPreProcessor",
        &format!("resized image: {:#?}", image.dimensions()),
    );

    let mut array = ndarray::Array::from_shape_fn(
        (1, channels as usize, height as usize, width as usize),
        |(_, c, j, i)| {
            let pixel = image.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        },
    );

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }
    // Ok(f32_array_to_bytes(array.as_slice().unwrap()).await)
    Ok(f32_array_to_bytes(array.as_slice().unwrap()))
}
