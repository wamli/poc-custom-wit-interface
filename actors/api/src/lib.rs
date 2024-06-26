wit_bindgen::generate!();

use urlencoding::decode; 
use serde_json::from_str; 
use std::io::{self, Write};
use http::StatusCode;
// use serde::{de::value, Deserialize, Serialize};
use wasi::http::types::*;
use crate::wasi::logging::logging::*;
use crate::wamli::ml::inference::predict;
use crate::wasi::io::streams::StreamError;
use crate::wamli::ml::conversion::convert;
use wamli::ml::{conversion::ConversionRequest};
use exports::wasi::http::incoming_handler::Guest;
use crate::wamli::ml::conversion::{Tensor, ValueType};


type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
const DIMENSIONS_PARAM_NAME: &str = "dimensions";
const VALUE_TYPE_PARAM_NAME: &str = "value_type";
// [1, 3, 244, 244] which is a typical ImageNet dimension
const DEFAULT_DIMENIONS: &str = "%5B1%2C3%2C224%2C224%5D"; 
const DEFAULT_VALUE_TYPE: &str = "F32";

struct Api;

#[derive(Debug)]
pub struct Error {
    status_code: StatusCode,
    message: String,
}

impl Error {
    fn body_parsing_error() -> Self {
        Error {
            status_code: StatusCode::BAD_REQUEST,
            message: format!("Could not extract attachment from request body"),
        }
    }

    fn body_deserialization_error(e: Error) -> Self {
        Error {
            status_code: StatusCode::BAD_REQUEST,
            message: format!("Error deserializing request body to tensor: {:?}", e),
        }
    }

    fn bad_request_parameter_value_type(e: Error) -> Self {
        Error {
            status_code: StatusCode::BAD_REQUEST,
            message: format!("Invalid value-type detected in request: {:?}", e),
        }
    }

    fn not_found() -> Self {
        Error {
            status_code: StatusCode::NOT_FOUND,
            message: "Object not found".to_string(),
        }
    }
}

impl Guest for Api {
    fn handle(req: IncomingRequest, response_out: ResponseOutparam) 
    {
        let path_and_query = req.path_with_query().unwrap_or_else(|| "/".to_string());
        
        let method = req.method();

        // Here's a quick reference for these characters in percent-encoding:
        //     [ is encoded as %5B
        //     ] is encoded as %5D
        //     , is encoded as %2C
        //
        // see RFC3986

        // Derive the appropriate model, dimensions and value-type from the path & query string
        //
        // ex. 'localhost:8080/mobilenetv27:latest?dimensions=[2,2]&value_type=F32'
        // ex. 'localhost:8080/mobilenetv27%3Alatest?dimensions=%5B2%2C2%5D&value_type=F32'
        // ex. 'localhost:8080/no-preprocessing/mobilenetv27%3Alatest?dimensions=%5B2%2C2%5D&value_type=F32'
        let (full_path, dimensions, value_type) = match path_and_query.split_once('?') 
        {
            Some((path, query)) => {
                // We have a query string, so let's split it into dimensions name and a value-type
                let dimensions = query
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .find(|(k, _)| *k == DIMENSIONS_PARAM_NAME)
                    .map(|(_, v)| v)
                    .unwrap_or(DEFAULT_DIMENIONS);

                let value_type = query
                    .split('&')
                    .skip(1)
                    .filter_map(|p| p.split_once('='))
                    .find(|(k, _)| *k == VALUE_TYPE_PARAM_NAME)
                    .map(|(_, v)| v)
                    .unwrap_or(DEFAULT_VALUE_TYPE);
                
                (path.trim_matches('/').to_string(), dimensions.to_string(), value_type.to_string())
            }
            None => (
                path_and_query.trim_matches('/').to_string(),
                DEFAULT_DIMENIONS.to_string(),
                DEFAULT_VALUE_TYPE.to_string(),
            ),
        };

        let decoded_dimensions = decode(&dimensions).unwrap_or(std::borrow::Cow::Borrowed(""));
        let dimensions_vector: Vec<u32> = from_str(&decoded_dimensions).unwrap_or(vec![]);

        let decoded_model = decode(&full_path).unwrap();

        let segments: Vec<&str> = decoded_model.trim_end_matches('/').split('/').collect();

        log(
            Level::Info, 
            "Api", 
            &format!(
                "INCOMING REQUEST - method: '{:?}' path: '{:?}' dimensions: '{:?}', value_type: '{}'", &method, &segments, &dimensions_vector, &value_type)
        );


        match (method, segments.as_slice()) {
            (Method::Get, ["prefetch", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing PREFETCH with model_id: '{:?}' ", model_id));
            },

            (Method::Put, [model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing WITH PRE- AND POST-PROCESSING, with model_id: '{:?}' ", model_id));

                send_positive_resonse(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
                return;
            },

            (Method::Put, ["no-preprocessing", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing NO-PREPROCESSING with model_id: '{:?}' ", model_id));

                send_positive_resonse(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
                return;
            },

            (Method::Put, ["no-postprocessing", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing NO-POSTPROCESSING with model_id: '{:?}' ", model_id));
            },

            (Method::Put, ["inference-only", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing INFERENCE-ONLY with model_id: '{:?}' ", model_id));

                let body = match parse_request_body(req) {
                    Ok(b) => b,
                    Err(error) => {
                        log(Level::Error, "Api", &format!("failed to deserialize the input tensor from body: {:?}", error));
                        send_response_error(
                            response_out,
                            error,
                        );
                        return;
                    }
                };

                let val_type:ValueType = match String::from(value_type).parse::<ValueType>(){
                    Ok(v) => v,
                    Err(error) => {
                        log(Level::Error, "Api", &format!("Invalid value-type detected in request: {:?}", error));
                    
                        send_response_error(
                            response_out,
                            Error::bad_request_parameter_value_type(error),
                        );
                        return;
                    },
                };

                let tensor = Tensor {
                    dimensions: dimensions_vector,
                    value_types: vec![val_type],
                    bit_flags: 0,
                    data: body,
                };

                let prediction = predict("Greeting from api!");
                log(Level::Info, "", &format!("-------> PREDICTION received: {:?}", prediction));
            },

            _ => {
                send_response_error(
                    response_out,
                    Error::not_found(),
                );
                return;
            }
        }

        // let input = InferenceInput {
        //     model: "model_name".to_string(),
        //     index: 0,
        //     tensor,
        // };

        // let conversion_request = ConversionRequest {
        //     tensor: tensor,
        //     target_dimensions: None,
        //     target_value: None,
        // };

        // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
        let interface1 = wasmcloud::bus::lattice::CallTargetInterface::new(
            "wamli",
            "ml",
            "conversion",
        );

        let interface2 = wasmcloud::bus::lattice::CallTargetInterface::new(
            "wamli",
            "ml",
            "conversion",
        );
        
        // wasmcloud::bus::lattice::set_link_name("preprocessor01", vec![interface1]);
        // let converted = convert(&conversion_request);
        // log(Level::Info, "Api", &format!("--------> CONVERSION received from PRE-processor: {:?}", converted));

        // wasmcloud::bus::lattice::set_link_name("postprocessor01", vec![interface2]);
        // let converted = convert(&conversion_request);
        // log(Level::Info, "Api", &format!("--------> CONVERSION received from POST-processor: {:?}", converted));

        send_positive_resonse(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
    }
}

// export! defines that the `Api` struct defined below is going to define
// the exports of the `world`, namely the `handle` function.
export!(Api);

fn send_positive_resonse(response_out: ResponseOutparam, content: &str) {
    let response = OutgoingResponse::new(Fields::new());
    response
        .set_status_code(200)
        .expect("Unable to set status code");
    
    let response_body = response.body().expect("body called more than once");
    let mut writer = response_body.write().expect("should only call write once");

    let mut stream = OutputStreamWriter::from(&mut writer);

    if let Err(e) = stream.write_all(content.as_bytes()) {
        log(
            Level::Error,
            "Api",
            format!("Failed to write to stream: {}", e).as_str(),
        );
        return;
    }
    // Make sure to release the write resources
    drop(writer);
    OutgoingBody::finish(response_body, None).expect("failed to finish response body");
    
    ResponseOutparam::set(response_out, Ok(response));
}

fn send_response_error(response_out: ResponseOutparam, error: Error) {
    let response = OutgoingResponse::new(Fields::new());
    response
        .set_status_code(error.status_code.as_u16())
        .expect("Unable to set status code");
    let response_body = response.body().expect("body called more than once");
    let mut writer = response_body.write().expect("should only call write once");

    let mut stream = OutputStreamWriter::from(&mut writer);

    if let Err(e) = stream.write_all(error.message.as_bytes()) {
        log(
            Level::Error,
            "handle",
            format!("Failed to write to stream: {}", e).as_str(),
        );
        return;
    }
    // Make sure to release the write resources
    drop(writer);
    OutgoingBody::finish(response_body, None).expect("failed to finish response body");
    
    ResponseOutparam::set(response_out, Ok(response));
}

fn parse_request_body(request: IncomingRequest) -> Result<Vec<u8>> {
    let body = match request.consume() {
        Ok(b) => Ok(b),
        Err(_) => Err(Error::body_parsing_error()),
    }?;

    let stream = body
        .stream()
        .expect("Unable to get stream from request body");

    let mut buffer: Vec<u8> = Vec::new();

    loop {
        // let mut chunk: Vec<u8> = Vec::new();
        if let Some(chunk) = stream.read(1024).ok() {
            log(
                Level::Info,
                "Api",
                format!("Read some bytes to buffer: {}", chunk.len()).as_str(),
            );

            if chunk.len() == 0 {
                // End of file reached
                break;
            }
            buffer.extend_from_slice(&chunk[..chunk.len()]);
        }
    }
        
    if buffer.len() == 0 {
        return Err(Error::body_parsing_error());
    }

    log(Level::Info, "Api", &format!("--------> Read body into buffer - total length: {:?}", buffer.len()));

    return Ok(buffer);
}

pub struct OutputStreamWriter<'a> {
    stream: &'a mut crate::wasi::io::streams::OutputStream,
}

impl<'a> From<&'a mut crate::wasi::io::streams::OutputStream> for OutputStreamWriter<'a> {
    fn from(stream: &'a mut crate::wasi::io::streams::OutputStream) -> Self {
        Self { stream }
    }
}

impl std::io::Write for OutputStreamWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = match self.stream.check_write().map(std::num::NonZeroU64::new) {
            Ok(Some(n)) => n,
            Ok(None) | Err(StreamError::Closed) => return Ok(0),
            Err(StreamError::LastOperationFailed(e)) => {
                return Err(io::Error::new(io::ErrorKind::Other, e.to_debug_string()))
            }
        };
        let n = n
            .get()
            .try_into()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let n = buf.len().min(n);
        self.stream.write(&buf[..n]).map_err(|e| match e {
            StreamError::Closed => io::ErrorKind::UnexpectedEof.into(),
            StreamError::LastOperationFailed(e) => {
                io::Error::new(io::ErrorKind::Other, e.to_debug_string())
            }
        })?;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream
            .blocking_flush()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// Implement the FromStr trait for the ValueType enum
impl std::str::FromStr for ValueType {
    type Err = Error;

    fn from_str(s: &str) -> Result<ValueType> {
        let binding = s.to_lowercase();
        let s_lower = binding.as_ref();
        match s_lower {
            "u8"  => Ok(ValueType::U8),
            "u16" => Ok(ValueType::U16),
            "u32" => Ok(ValueType::U32),
            "u64" => Ok(ValueType::U64),
            "u128"=> Ok(ValueType::U128),
            "s8"  => Ok(ValueType::S8),
            "s16" => Ok(ValueType::S16),
            "s32" => Ok(ValueType::S32),
            "s64" => Ok(ValueType::S64),
            "s128"=> Ok(ValueType::S128),
            "f16" => Ok(ValueType::F16),
            "f32" => Ok(ValueType::F32),
            "f64" => Ok(ValueType::F64),
            "f128"=> Ok(ValueType::F128),
            _ => Err(Error {
                    status_code: StatusCode::BAD_GATEWAY,
                    message: format!("Error when communicating with blobstore"),
                }),
        }
    }
}