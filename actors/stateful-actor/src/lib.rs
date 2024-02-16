wit_bindgen::generate!({
    world: "interfaces",
    exports: {
        "wamli:ai/model": ActorState,
    },
});

use exports::wamli::ai::model::{self, ExecutionTarget, Metadata};

struct ActorState;

impl model::Guest for ActorState {
    fn get_data() -> Result<Vec<u8>, model::Error>{
        Ok(vec![])
    }

    fn get_metadata() -> Result<Metadata, model::Error>{
        let md = Metadata {
            model_name: "mobilenetv2-7".to_string(),
            graph_encoding: "onnx".to_string(),
            execution_target: ExecutionTarget::Cpu,
            tensor_type: model::TensorType::F32,
            tensor_dimensions_in: vec![1, 3, 224, 224],
            tensor_dimensions_out: vec![1, 1000, 1, 1],
        };
        
        Ok(md)
    }
}
