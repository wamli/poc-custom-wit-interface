// use ndarray::Data;

use crate::processor::preprocess;
use crate::wamli::ml::types::Tensor;
use crate::wamli::ml::types::MlError;
use crate::wasi::logging::logging::*;
use crate::exports::wamli::ml::conversion::{DataType, Guest, Dimensions};

mod processor;

wit_bindgen::generate!();

const BATCH: u32 = 1;
const CHANNELS: u32 = 3;
const HEIGHT: u32 = 224;
const WIDTH: u32 = 224;

const SHAPE: [u32;4] = [BATCH, CHANNELS, HEIGHT, WIDTH];
const DTYPE: DataType = DataType::F32;

const WRONG_TENSOR_SHAPE: &str = "Expecting tensors of shape [1, 3, 224, 224]";
const WRONG_DTYPE: &str = "Expecting tensors of dtype F32";

struct ImagenetPreProcessor;

impl Guest for ImagenetPreProcessor {
    fn convert(tensor: Tensor, _to_shape: Option<Dimensions>, _to_dtype: Option<DataType>) -> Result<Tensor, MlError> {
        log(
            Level::Info,
            "ImagenetPreProcessor",
            "received conversion request",
        );

        let tensor_data: Vec<u8> = tensor.data;
        let tensor_shape: Vec<u32> = tensor.shape;
        let tensor_dtype: DataType = tensor.dtype;

        if tensor_dtype != DTYPE {
            log(
                Level::Warn,
                "ImagenetPreProcessor",
                WRONG_DTYPE,
            );
        }

        if tensor_shape != SHAPE.to_vec() {
            log(
                Level::Error,
                "ImagenetPreProcessor",
                WRONG_TENSOR_SHAPE,
            );

            return Err(MlError::Internal(WRONG_TENSOR_SHAPE.to_string()));
        }

        log(
            Level::Info,
            "ImagenetPreProcessor",
            "going to enter preprocess()",
        );

        // let converted_tensor_data = match preprocess(&tensor_data, CHANNELS, HEIGHT, WIDTH) {
        //     Ok(t)      => t,
        //     Err(error) => {
        //         return Err(MlError::Internal(error.to_string()));
        //     }
        // };

        let v_size:usize = 610_000;

        let mut converted_tensor_data: Vec<u8> = Vec::with_capacity(v_size);

        // These are all done without reallocating...
        for i in 0..v_size {
            converted_tensor_data.push((i % 255) as u8);
        }

        let tensor_length = converted_tensor_data.len();

        let tensor_out = Tensor {
            shape: vec![1, CHANNELS, HEIGHT, WIDTH],
            dtype: DataType::F32,
            data: converted_tensor_data,
        };

        log(
            Level::Info,
            "ImagenetPreProcessor",
            &format!("leaving preprocess(), returning tensor of length {}", tensor_length),
        );

        // let tensor_out = Tensor {
        //     shape: vec![1],
        //     dtype: DataType::F32,
        //     data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9]
        // };

        Ok(tensor_out)
    }
}

export!(ImagenetPreProcessor);
