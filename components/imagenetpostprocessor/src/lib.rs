//! This actor is designed to support the following two models:
//! * https://tfhub.dev/tensorflow/coral-model/mobilenet_v1_1.0_224_quantized/1/default/1
//!     - additionally, see the [tflite labels](https://github.com/google-coral/edgetpu/blob/master/test_data/imagenet_labels.txt)
//! * https://github.com/onnx/models/tree/main/vision/classification/mobilenet
//!     - additionally, see the [ONNX labels](https://github.com/onnx/models/blob/main/vision/classification/synset.txt)

use std::io::Cursor;
use ndarray::{Array, ArrayBase};
use wamli::ml::types::Classification;
use crate::wasi::logging::logging::*;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::exports::wamli::ml::classification::{Guest, Tensor, Classifications, MlError};

mod imagenet_labels_onnx;
mod imagenet_labels_tflite;

pub type Result<T> = std::io::Result<T>;

wit_bindgen::generate!();

struct ImagenetPostProcessor;

impl Guest for ImagenetPostProcessor {
    fn classify(tensor: Tensor) -> std::result::Result<Classifications, MlError> {
        log(Level::Info, "ImagenetPostProcessor", "received conversion request");
        
        let tensor_data = tensor.data;

        let raw_result_f32 = match bytes_to_f32_vec(tensor_data) {
            Ok(rd) => rd,
            Err(error) => {
                return Err(MlError::Processor(error.to_string()));
            }
        };

        let labels: Vec<String> = match raw_result_f32.len() {
            1000 => { imagenet_labels_onnx::IMAGENT_LABELS_ONNX.lines().map(String::from).collect() },
               _ => { imagenet_labels_tflite::IMAGENT_LABELS_TFLITE.lines().map(String::from).collect() },
        };

        let probabilities: Vec<(usize, f32)> = match raw_result_f32.len() {
            1000 => { get_onnx_probabilities(raw_result_f32) },
            1001 => { get_tflite_probabilities(raw_result_f32) },
               _ => { vec![(1111, -1.0); 5]},
        };

        let mut classifications: Vec<Classification> = Vec::new();

        for i in 0..5 {
            let classification = Classification {
                label: labels[probabilities[i].0].clone(),
                probability: probabilities[i].1,
            };

            classifications.push(classification);
        }

        Ok(classifications)
    }
}

export!(ImagenetPostProcessor);

// pub async fn bytes_to_f32_vec(data: Vec<u8>) -> Result<Vec<f32>> {
pub fn bytes_to_f32_vec(data: Vec<u8>) -> Result<Vec<f32>> {
    data.chunks(4)
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_f32::<LittleEndian>()
        })
        .collect()
}

// pub async fn get_tflite_probabilities(raw_result: std::vec::Vec<f32>) -> Vec<(usize, f32)> {
pub fn get_tflite_probabilities(raw_result: std::vec::Vec<f32>) -> Vec<(usize, f32)> {
    let output_tensor = Array::from_shape_vec((1, 1001, 1, 1), raw_result).unwrap();
    
    let mut probabilities: Vec<(usize, f32)> = output_tensor
    .into_iter()
    .enumerate()
    .collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    return probabilities;
}

// pub async fn get_onnx_probabilities(raw_result: std::vec::Vec<f32>) -> Vec<(usize, f32)> {
pub fn get_onnx_probabilities(raw_result: std::vec::Vec<f32>) -> Vec<(usize, f32)> {
    let output_tensor = Array::from_shape_vec((1, 1000, 1, 1), raw_result).unwrap();

    let mut probabilities: Vec<(usize, f32)> = output_tensor
        .softmax(ndarray::Axis(1))
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    return probabilities;
}

pub trait NdArrayTensor<S, T, D> {
    /// https://en.wikipedia.org/wiki/Softmax_function)
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
        <S as ndarray::RawData>::Elem: std::clone::Clone,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign;
}

impl<S, T, D> NdArrayTensor<S, T, D> for ArrayBase<S, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
    <S as ndarray::RawData>::Elem: std::clone::Clone,
    T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
{
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D> {
        let mut new_array: Array<T, D> = self.to_owned();
        new_array.map_inplace(|v| *v = v.exp());
        let sum = new_array.sum_axis(axis).insert_axis(axis);
        new_array /= &sum;

        new_array
    }
}