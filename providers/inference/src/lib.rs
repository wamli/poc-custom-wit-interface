pub mod config;
pub mod engine;
pub mod provider;
pub mod data_loader;

wit_bindgen_wrpc::generate!();

pub use crate::exports::wamli::ml::inference::Handler;
pub use crate::wamli::ml::types::{DataType, MlError, Tensor};

