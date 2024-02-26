wit_bindgen::generate!({
    world: "api",
    exports: {
        "wasi:http/incoming-handler": HttpServer,
    },
});

use exports::wasi::http::incoming_handler::Guest;
use crate::wamli::ml::inference::{predict, InferenceInput};
use crate::wamli::ml::types::{Tensor, ValueType, ValueTypes};
use wasi::http::types::*;

struct HttpServer;
// struct MlInference;

impl Guest for HttpServer {
    fn handle(_request: IncomingRequest, response_out: ResponseOutparam) {

        let tensor = Tensor {
            dimensions: vec![],
            value_types: vec![ValueType::U16],
            bit_flags: 0,
            data: vec![],
        };

        let input = InferenceInput {
            model: "model_name".to_string(),
            index: 0,
            tensor,
        };
    
        // do whatever you have to do here!
        predict(&input);


        let response = OutgoingResponse::new(Fields::new());
        response.set_status_code(200).unwrap();
        let response_body = response.body().unwrap();
        response_body
            .write()
            .unwrap()
            .blocking_write_and_flush(b"Hello from Rust!\n")
            .unwrap();
        OutgoingBody::finish(response_body, None).expect("failed to finish response body");
        ResponseOutparam::set(response_out, Ok(response));
    }
}
