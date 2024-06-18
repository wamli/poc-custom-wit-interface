mod ai_model;

use ai_model::MODEL;
use exports::wamli::ml::model::{self, ExecutionTarget, Metadata, TensorType};

wit_bindgen::generate!({
    world: "interfaces",
    exports: {
        "wamli:ml/model": AiModel,
    },
});

struct AiModel;

impl model::Guest for AiModel {
    fn get_data() -> Result<Vec<u8>, model::Error>{
        Ok(MODEL.to_vec())
    }

    fn get_metadata() -> Result<Metadata, model::Error>{
        let md = Metadata {
            model_name: "mobilenetv2-7".to_string(),
            graph_encoding: "onnx".to_string(),
            execution_target: ExecutionTarget::Cpu,
            tensor_type: TensorType::F32,
            tensor_dimensions_in: vec![1, 3, 224, 224],
            tensor_dimensions_out: vec![1, 1000, 1, 1],
        };
        
        Ok(md)
    }
}
