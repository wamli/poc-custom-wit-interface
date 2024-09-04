use urlencoding::decode; 
use serde_json::from_str; 
use std::io::{Read, Write};
use crate::wasi::logging::logging::*;
use crate::streams::InputStreamReader;

use crate::Api;
use crate::HttpGuest;

use crate::wasmcloud::bus::lattice::{CallTargetInterface, set_link_name};
use crate::errors::{Result, Error, send_positive_response, send_response_error};

use crate::{
    wasi::http::types::*,
    wamli::ml::{
        types::{DataType, Tensor},
        inference::{prefetch, preempt, predict},
        conversion::convert,
        classification::classify,
    },
};

#[allow(dead_code)]
const DIMENSIONS_PARAM_NAME: &str = "dimensions";
const VALUE_TYPE_PARAM_NAME: &str = "value_type";
// [1, 3, 244, 244] which is a typical ImageNet dimension
const DEFAULT_DIMENIONS: &str = "%5B1%2C3%2C224%2C224%5D"; 
const DEFAULT_VALUE_TYPE: &str = "NA";

const PREPROCESSOR:  &str = "imagenetpreprocessor";
const POSTPROCESSOR: &str = "imagenetpostprocessor";

impl HttpGuest for Api {
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
            (Method::Delete, [model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: going to DELETE model '{:?}' ", model_id));

                let preempt_result = preempt(model_id);

                log(Level::Info, "Api", &format!("--------> API: PREEMPT result: '{:?}' ", preempt_result));
                
                send_positive_response(response_out, &format!("Feedback from inference provider: {:?}", preempt_result));
                return;
            },

            (Method::Get, ["prefetch", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing PREFETCH with model_id: '{:?}' ", model_id));

                let prefetch_result = prefetch(model_id);

                log(Level::Info, "Api", &format!("--------> API: PREFETCH result: '{:?}' ", prefetch_result));
                
                send_positive_response(response_out, &format!("Feedback from inference provider: {:?}", prefetch_result));
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

                // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
                let interface1 = CallTargetInterface::new(
                    "wamli",
                    "ml",
                    "conversion",
                );
                
                set_link_name(PREPROCESSOR, vec![interface1]);
                
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
                
                log(Level::Info, "Api", &format!("--------> CONVERSION of length '{}' bytes received from PRE-processor", converted.data.len()));

                let prediction = match predict(model_id, &converted) {
                    Ok(t) => t,
                    Err(error) => {
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };

                log(Level::Info, "", &format!("-------> PREDICTION received of length: '{}' bytes, ", prediction.data.len()));

                let interface2 = CallTargetInterface::new(
                    "wamli",
                    "ml",
                    "classification",
                );

                set_link_name(POSTPROCESSOR, vec![interface2]);
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
                        log(Level::Info, "Api", &format!("--------> CLASSIFICATION yields an error: {:?}", &error));
                        send_response_error(
                            response_out,
                            Error::internal_server_error(error),
                        );
                        return;
                    },
                };

                log(Level::Info, "Api", &format!("--------> CONVERSION received from POST-processor: {:?}", &classifications));


                send_positive_response(response_out, &format!("{:?}", classifications));
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
                let interface1 = CallTargetInterface::new(
                    "wamli",
                    "ml",
                    "conversion",
                );
                
                set_link_name(PREPROCESSOR, vec![interface1]);
                
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
                
                log(Level::Info, "Api", &format!("--------> CONVERSION received from PRE-processor of length: {:?}", converted.data.len()));

                send_positive_response(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
                return;
            },

            (Method::Put, ["no-preprocessing", model_id]) => {
                log(Level::Info, "Api", &format!("--------> API: executing NO-PREPROCESSING with model_id: '{:?}' ", model_id));

                send_positive_response(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
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

        send_positive_response(response_out, "HAPPY TO SEE YOU WENT DOWN THE LUCKY PATH!");
    }
}

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

// export! defines that the `Api` struct defined below is going to define
// the exports of the `world`, namely the `handle` function.
// export!(Api);