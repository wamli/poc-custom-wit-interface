wit_bindgen::generate!();

use exports::wasi::http::incoming_handler::Guest;
use wamli::ml::conversion::ConversionRequest;
use crate::wamli::ml::conversion::{Tensor, ValueType};
use crate::wasi::logging::logging::*;
use crate::wamli::ml::inference::predict;
use crate::wamli::ml::conversion::convert;
use wasi::http::types::*;

struct Api;

impl Guest for Api {
    fn handle(_request: IncomingRequest, response_out: ResponseOutparam) {

        let tensor = Tensor {
            dimensions: vec![],
            value_types: vec![ValueType::U16],
            bit_flags: 0,
            data: vec![],
        };

        // let input = InferenceInput {
        //     model: "model_name".to_string(),
        //     index: 0,
        //     tensor,
        // };

        let conversion_request = ConversionRequest {
            tensor: tensor,
            target_dimensions: None,
            target_value: None,
        };

        // https://wasmcloud.slack.com/archives/CS38R7N9Y/p1719256911613509?thread_ts=1718986484.246259&cid=CS38R7N9Y
        let interface = wasmcloud::bus::lattice::CallTargetInterface::new(
            "wamli",
            "ml",
            "conversion",
        );
        
        wasmcloud::bus::lattice::set_link_name("preprocessor01", vec![interface]);


        let converted = convert(&conversion_request);
        log(Level::Info, "Api", &format!("--------> CONVERSION received: {:?}", converted));


        let prediction = predict("Greeting from api!");
        log(Level::Info, "", &format!("-------> PREDICTION received: {:?}", prediction));

        let response = OutgoingResponse::new(Fields::new());
        response.set_status_code(200).unwrap();
        let response_body = response.body().unwrap();
        response_body
            .write()
            .unwrap()
            .blocking_write_and_flush(b"Hello from Rusty crab!\n")
            .unwrap();
        OutgoingBody::finish(response_body, None).expect("failed to finish response body");
        ResponseOutparam::set(response_out, Ok(response));
    }
}

// export! defines that the `Api` struct defined below is going to define
// the exports of the `world`, namely the `handle` function.
export!(Api);
