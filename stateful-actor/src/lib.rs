wit_bindgen::generate!({
    world: "interfaces",
    exports: {
        "wamli:stateful/state": ActorState,
    },
});

use exports::wamli::stateful::state::{self};

struct ActorState;

impl state::Guest for ActorState {
    fn get_data() -> Result<Vec<u8>, state::Error>{
        Ok(vec![])
    }

    fn get_metadata() -> Result<Vec<String>, state::Error>{
        Ok(vec![])
    }
}
