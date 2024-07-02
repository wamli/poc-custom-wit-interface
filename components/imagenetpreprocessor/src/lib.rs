use crate::processor::preprocess;
use crate::wamli::ml::types::Tensor;
use crate::wamli::ml::types::MlError;
use crate::wasi::logging::logging::*;
use crate::exports::wamli::ml::conversion::{DataType, Guest, Dimensions};

mod processor;

wit_bindgen::generate!();

const CHANNELS: u32 = 3;
const HEIGHT: u32 = 224;
const WIDTH: u32 = 224;

const SHAPE: [u32;3] = [CHANNELS, HEIGHT, WIDTH];
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

        let converted_tensor_data = match preprocess(&tensor_data, CHANNELS, HEIGHT, WIDTH) {
            Ok(t)      => t,
            Err(error) => {
                return Err(MlError::Internal(error.to_string()));
            }
        };

        let tensor_out = Tensor {
            shape: vec![1, CHANNELS, HEIGHT, WIDTH],
            dtype: DataType::F32,
            data: converted_tensor_data,
        };

        Ok(tensor_out)
    }
}

export!(ImagenetPreProcessor);
