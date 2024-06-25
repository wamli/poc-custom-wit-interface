use wamli::ml::types::Status;
use crate::wamli::ml::types::Tensor;
use crate::wasi::logging::logging::*;
use crate::exports::wamli::ml::conversion::Guest;
use exports::wamli::ml::conversion::ConversionResponse;
use exports::wamli::ml::conversion::ConversionRequest;

wit_bindgen::generate!();

struct PostProcessor;

impl Guest for PostProcessor {
    fn convert(_input: ConversionRequest) -> ConversionResponse {
        log(Level::Info, "PostProcessor_01", "------ POST-PROCESSOR RECEIVED CONVERSION REQUEST ------");
        
        let tensor = Tensor {
            dimensions: vec![1, 2, 3],
            value_types: vec![],
            bit_flags: 0,
            data: vec![],
        };

        ConversionResponse { status: Status::Success(true), tensor: tensor }
    }
}

export!(PostProcessor);