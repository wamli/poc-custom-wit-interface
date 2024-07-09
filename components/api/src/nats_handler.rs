use crate::NatsGuest;
use crate::wasi::logging::logging::*;
use crate::wasmcloud::messaging::types;
use crate::wasmcloud::messaging::consumer;

use crate::Api;

impl NatsGuest for Api {
    fn handle_message(msg: types::BrokerMessage) -> Result<(), String> {
        if let Some(reply_to) = msg.reply_to {
            let body_size = msg.body.len();
            log(Level::Info, "Api", &format!("--------> Read NATS request body with length: {:?}", body_size));

            consumer::publish(&types::BrokerMessage {
                subject: reply_to,
                reply_to: None,
                body: msg.body,
            })
        } else {
            log(
                Level::Warn,
                "",
                "No reply_to field in message, ignoring message",
            );
            Ok(())
        }
    }
}

// export!(Api);