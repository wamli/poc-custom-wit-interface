use http::StatusCode;
use std::io::Write;
use crate::wasi::logging::logging::*;
use crate::streams::OutputStreamWriter;
use crate::{
    wasi::http::types::*,
    wamli::ml::types::{DataType, MlError}
};
use crate::wasi::http::types::{OutgoingBody, ResponseOutparam, OutgoingResponse};


pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    status_code: StatusCode,
    message: String,
}

impl Error {
    pub fn body_parsing_error() -> Self {
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

    pub fn bad_request_parameter_value_type(e: Error) -> Self {
        Error {
            status_code: StatusCode::BAD_REQUEST,
            message: format!("Invalid value-type detected in request: {:?}", e),
        }
    }

    pub fn internal_server_error(e: MlError) -> Self {
        Error {
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("Internal server error: {:?}", e),
        }
    }

    pub fn not_found() -> Self {
        Error {
            status_code: StatusCode::NOT_FOUND,
            message: "Object not found".to_string(),
        }
    }
}

pub fn send_positive_response(response_out: ResponseOutparam, content: &str) {
    let response = OutgoingResponse::new(Fields::new());
    response.set_status_code(200).expect("Unable to set status code");
    
    let response_body = response.body().expect("body called more than once");

    ResponseOutparam::set(response_out, Ok(response));

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
}

pub fn send_response_error(response_out: ResponseOutparam, error: Error) {
    log(Level::Error, "Api", &format!("Failed to process request: {:?}", error));
    
    let response = OutgoingResponse::new(Fields::new());
    response
        .set_status_code(error.status_code.as_u16())
        .expect("Unable to set status code");
    let response_body: OutgoingBody = response.body().expect("body called more than once");

    ResponseOutparam::set(response_out, Ok(response));

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