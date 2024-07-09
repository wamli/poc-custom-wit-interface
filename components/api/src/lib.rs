wit_bindgen::generate!();

use http::StatusCode;
use urlencoding::decode; 
use wasi::http::types::*;
use serde_json::from_str; 
use std::io::{Read, Write};
use wamli::ml::inference::prefetch;
use crate::wasi::logging::logging::*;
use crate::wamli::ml::inference::predict;
use exports::wasi::http::incoming_handler::Guest;
use crate::wamli::ml::conversion::convert;
use crate::wamli::ml::types::{Tensor, DataType, MlError};
use crate::wamli::ml::classification::classify;


type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
const DIMENSIONS_PARAM_NAME: &str = "dimensions";
const VALUE_TYPE_PARAM_NAME: &str = "value_type";
// [1, 3, 244, 244] which is a typical ImageNet dimension
const DEFAULT_DIMENIONS: &str = "%5B1%2C3%2C224%2C224%5D"; 
const DEFAULT_VALUE_TYPE: &str = "NA";

const PREPROCESSOR:  &str = "imagenetpreprocessor";
const POSTPROCESSOR: &str = "imagenetpostprocessor";

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

    // fn body_deserialization_error(e: Error) -> Self {
    //     Error {
    //         status_code: StatusCode::BAD_REQUEST,
    //         message: format!("Error deserializing request body to tensor: {:?}", e),
    //     }
    // }

    fn bad_request_parameter_value_type(e: Error) -> Self {
        Error {
            status_code: StatusCode::BAD_REQUEST,
            message: format!("Invalid value-type detected in request: {:?}", e),
        }
    }

    fn internal_server_error(e: MlError) -> Self {
        Error {
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("Internal server error: {:?}", e),
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
        // ex. 'localhost:8081/mobilenetv27:latest?dimensions=[2,2]&value_type=F32'
        // ex. 'localhost:8081/mobilenetv27%3Alatest?dimensions=%5B2%2C2%5D&value_type=F32'
        // ex. 'localhost:8081/no-preprocessing/mobilenetv27%3Alatest?dimensions=%5B2%2C2%5D&value_type=F32'
        // ex. 'localhost:8081/prefetch/wamli-mobilenetv27%3Alatest?dimensions=%5B2%2C2%5D&value_type=F32'
        // ex. 'localhost:8081/prefetch/wamli-mobilenetv27%3Alatest'
        let (full_path, dimensions, value_type) = match path_and_query.split_once('?') 
        {
            Some((path, query)) => {
                // We have a query string, so let's split it into dimensions name and a value-type
                let dimensions = query
                    .split('&')
                    .filter_map(|p| p.split_once('='))
                    .find(|(k, _)| *k == DIMENSIONS_PARAM_NAME)
                    .map(|(_, v)| v);
                    // .unwrap_or(DEFAULT_DIMENIONS);

                let value_type = query
                    .split('&')
                    .skip(1)
                    .filter_map(|p| p.split_once('='))
                    .find(|(k, _)| *k == VALUE_TYPE_PARAM_NAME)
                    .map(|(_, v)| v)
                    .unwrap_or(DEFAULT_VALUE_TYPE);
                
                // (path.trim_matches('/').to_string(), dimensions.unwrap_or("").to_string(), value_type.unwrap_or("").to_string())
                (path.trim_matches('/').to_string(), dimensions, value_type.to_string())
            }
            None => (
                path_and_query.trim_matches('/').to_string(),
                Some(DEFAULT_DIMENIONS),
                DEFAULT_VALUE_TYPE.to_string(),
            ),
        };

        // dimensions.and_then(f);

        let dimensions_vector: Option<Vec<u32>> = dimensions.and_then(|d| {
            let decoded_dimensions = decode(&d).unwrap_or(std::borrow::Cow::Borrowed(""));
            from_str(&decoded_dimensions).ok()
        });

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

                let prefetch_result = prefetch(model_id);

                log(Level::Info, "Api", &format!("--------> API: PREFETCH result: '{:?}' ", prefetch_result));
                
                send_positive_resonse(response_out, &format!("Feedback from inference provider: {:?}", prefetch_result));
                return;
            },

            (Method::Put, [model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing WITH PRE- AND POST-PROCESSING, with model_id: '{:?}' ", model_id));

                let body = match parse_request_body(req) {
                    Ok(b) => {
                        log(Level::Info, "Api", &format!("--------> Read http request body into buffer - total length: {:?}", b.len()));
                        b
                    },

                    Err(error) => {
                        send_response_error(
                            response_out,
                            error,
                        );
                        return;
                    }
                };

                let tensor = Tensor {
                    shape: vec![1, 3, 224, 224],
                    dtype: DataType::F32,
                    data: body,
                };

                // // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
                // let interface1 = wasmcloud::bus::lattice::CallTargetInterface::new(
                //     "wamli",
                //     "ml",
                //     "conversion",
                // );
                
                // wasmcloud::bus::lattice::set_link_name(PREPROCESSOR, vec![interface1]);
                
                let converted = match convert(&tensor, None, None) {
                    Ok(t) => t,
                    Err(error) => {
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };
                
                log(Level::Info, "Api", &format!("--------> CONVERSION received from PRE-processor: {:?}", converted));

                let prediction = match predict(model_id, &tensor) {
                    Ok(t) => t,
                    Err(error) => {
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };

                log(Level::Info, "", &format!("-------> PREDICTION received: {:?}", prediction));

                // let interface2 = wasmcloud::bus::lattice::CallTargetInterface::new(
                //     "wamli",
                //     "ml",
                //     "conversion",
                // );

                // wasmcloud::bus::lattice::set_link_name(POSTPROCESSOR, vec![interface2]);
                // let converted = match convert(&prediction, None, None) {
                //     Ok(t) => t,
                //     Err(error) => {
                //         send_response_error(
                //             response_out,
                //             Error::internal_server_error(error),
                //         );
                //         return;
                //     },
                // };

                let classifications = match classify(&prediction) {
                    Ok(classifications) => classifications,
                    Err(error) => {
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };

                log(Level::Info, "Api", &format!("--------> CONVERSION received from POST-processor: {:?}", &classifications));


                send_positive_resonse(response_out, &format!("{:?}", classifications));
                return;
            },

            (Method::Put, ["preprocessing-only", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: PREPROCESSING ONLY with model_id: '{:?}' ", model_id));

                let body = match parse_request_body(req) {
                    Ok(b) => {
                        log(Level::Info, "Api", &format!("--------> Read http request body into buffer - total length: {:?}", b.len()));
                        b
                    },

                    Err(error) => {
                        send_response_error(
                            response_out,
                            error,
                        );
                        return;
                    }
                };

                let tensor = Tensor {
                    shape: vec![1, 3, 224, 224],
                    dtype: DataType::F32,
                    data: body,
                };

                // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
                let interface1 = wasmcloud::bus::lattice::CallTargetInterface::new(
                    "wamli",
                    "ml",
                    "conversion",
                );
                
                wasmcloud::bus::lattice::set_link_name(PREPROCESSOR, vec![interface1]);
                
                let converted = match convert(&tensor, None, None) {
                    Ok(t) => t,
                    Err(error) => {
                        log(Level::Info, "Api", &format!("--------> CONVERSION received from PRE-processor went WRONG"));
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };
                
                log(Level::Info, "Api", &format!("--------> CONVERSION received from PRE-processor: {:?}", converted));

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

                // let body = get_body(req, response_out);
                let body = match parse_request_body(req) {
                    Ok(b) => {
                        log(Level::Info, "Api", &format!("--------> Read http request body into buffer - total length: {:?}", b.len()));
                        b
                    },

                    Err(error) => {
                        send_response_error(
                            response_out,
                            error,
                        );
                        return;
                    }
                };

                let val_type:DataType = match String::from(value_type).parse::<DataType>(){
                    Ok(v) => v,
                    Err(error) => {
                        send_response_error(
                            response_out,
                            Error::bad_request_parameter_value_type(error),
                        );
                        return;
                    },
                };

                let tensor = Tensor {
                    shape: dimensions_vector.unwrap_or(vec![]),
                    dtype: val_type,
                    data: body,
                };

                let prediction = predict(model_id, &tensor);
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

        // // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
        // let interface1 = wasmcloud::bus::lattice::CallTargetInterface::new(
        //     "wamli",
        //     "ml",
        //     "conversion",
        // );

        // let interface2 = wasmcloud::bus::lattice::CallTargetInterface::new(
        //     "wamli",
        //     "ml",
        //     "conversion",
        // );
        
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
    log(Level::Error, "Api", &format!("Failed to process request: {:?}", error));
    
    let response = OutgoingResponse::new(Fields::new());
    response
        .set_status_code(error.status_code.as_u16())
        .expect("Unable to set status code");
    let response_body: OutgoingBody = response.body().expect("body called more than once");
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

// fn get_body(request: IncomingRequest, response_out: ResponseOutparam) -> Vec<u8> {
//     match parse_request_body(request) {
//         Ok(b) => {
//             log(Level::Info, "Api", &format!("--------> Read http request body into buffer - total length: {:?}", b.len()));
//             b
//         },

//         Err(error) => {
//             log(Level::Error, "Api", &format!("Failed to deserialize the input tensor from body: {:?}", error));
//             send_response_error(
//                 response_out,
//                 error,
//             );
//             return vec![];
//         }
//     }
// }

fn parse_request_body(request: IncomingRequest) -> Result<Vec<u8>> {
    let body = match request.consume() {
        Ok(b) => Ok(b),
        Err(_) => Err(Error::body_parsing_error()),
    }?;

    let mut stream = body
        .stream()
        .expect("Unable to get stream from request body");

    let mut reader: InputStreamReader = (&mut stream).into();

    let mut buffer = Vec::<u8>::new();

    loop {
        let mut chunk = [0u8; 4096]; // Buffer to store the read data
        if let Some(no_bytes_read) = reader.read(&mut chunk).ok() {
            if no_bytes_read == 0 {
                // End of file reached
                break;
            }
            buffer.extend_from_slice(&chunk);
        }
    }
        
    if buffer.len() == 0 {
        return Err(Error::body_parsing_error());
    }

    return Ok(buffer);
}

// Implement the FromStr trait for the DataType enum
impl std::str::FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<DataType> {
        let binding = s.to_lowercase();
        let s_lower = binding.as_ref();
        match s_lower {
            "u8"  => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "u128"=> Ok(DataType::U128),
            "s8"  => Ok(DataType::S8),
            "s16" => Ok(DataType::S16),
            "s32" => Ok(DataType::S32),
            "s64" => Ok(DataType::S64),
            "s128"=> Ok(DataType::S128),
            "f16" => Ok(DataType::F16),
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),
            "f128"=> Ok(DataType::F128),
            "na"  => Ok(DataType::Na),
            _ => Err(Error {
                    status_code: StatusCode::BAD_REQUEST,
                    message: format!("Provided tensor's dtype is invalid"),
                }),
        }
    }
}

pub struct InputStreamReader<'a> {
    stream: &'a mut crate::wasi::io::streams::InputStream,
}

impl<'a> From<&'a mut crate::wasi::io::streams::InputStream> for InputStreamReader<'a> {
    fn from(stream: &'a mut crate::wasi::io::streams::InputStream) -> Self {
        Self { stream }
    }
}

impl std::io::Read for InputStreamReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use crate::wasi::io::streams::StreamError;
        use std::io;

        let n = buf
            .len()
            .try_into()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        match self.stream.blocking_read(n) {
            Ok(chunk) => {
                let n = chunk.len();
                if n > buf.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "more bytes read than requested",
                    ));
                }
                buf[..n].copy_from_slice(&chunk);
                Ok(n)
            }
            Err(StreamError::Closed) => Ok(0),
            Err(StreamError::LastOperationFailed(e)) => {
                Err(io::Error::new(io::ErrorKind::Other, e.to_debug_string()))
            }
        }
    }
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
        use crate::wasi::io::streams::StreamError;
        use std::io;

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

pub struct StdioStream<'a> {
    stdin: std::io::StdinLock<'a>,
    stdout: std::io::StdoutLock<'a>,
}

impl StdioStream<'_> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Read for StdioStream<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stdin.read(buf)
    }
}

impl Write for StdioStream<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stdout.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stdout.flush()
    }
}

impl Default for StdioStream<'_> {
    fn default() -> Self {
        Self {
            stdin: std::io::stdin().lock(),
            stdout: std::io::stdout().lock(),
        }
    }
}

#[cfg(feature = "futures")]
impl futures::AsyncRead for StdioStream<'_> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::task::Poll::Ready(self.stdin.read(buf))
    }
}

#[cfg(feature = "futures")]
impl futures::AsyncWrite for StdioStream<'_> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::task::Poll::Ready(self.stdout.write(buf))
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::task::Poll::Ready(self.stdout.flush())
    }

    fn poll_close(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.poll_flush(cx)
    }
}

#[cfg(feature = "tokio")]
impl tokio::io::AsyncRead for StdioStream<'_> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let mut fill = vec![0; buf.capacity()];
        std::task::Poll::Ready({
            let n = self.stdin.read(&mut fill)?;
            buf.put_slice(&fill[..n]);
            Ok(())
        })
    }
}

#[cfg(feature = "tokio")]
impl tokio::io::AsyncWrite for StdioStream<'_> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<Result<usize, std::io::Error>> {
        std::task::Poll::Ready(self.stdout.write(buf))
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        std::task::Poll::Ready(self.stdout.flush())
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), std::io::Error>> {
        self.poll_flush(cx)
    }
}