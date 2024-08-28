pub mod config;
pub mod data_loader;
pub mod engine;
pub mod provider;

wit_bindgen_wrpc::generate!({
    with: {
        "wamli:ml/types": generate,
        "wamli:ml/inference": generate,
    }
});

pub use crate::exports::wamli::ml::inference::Handler;
pub use crate::wamli::ml::types::{DataType, MlError, Tensor};
