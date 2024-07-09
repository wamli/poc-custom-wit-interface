use crate::engine::{
    DataType, ExecutionTarget, Graph, GraphEncoding, GraphExecutionContext, InferenceEngine,
    InferenceError, InferenceResult, Tensor,
};
use anyhow::Context;
use async_trait::async_trait;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array;
use std::{
    collections::{btree_map::Keys, BTreeMap},
    io::Cursor,
};
use tokio::sync::RwLock;
use tract_data::internal::tract_smallvec::SmallVec;
use tract_onnx::prelude::TValue; //__CB__NEW
use tract_onnx::{
    prelude::{Graph as TractGraph, Tensor as TractTensor, *},
    tract_hir::infer::InferenceOp,
};
use tract_tensorflow;

// use wasmcloud_interface_mlinference::{
//    InferenceOutput, Status, Tensor, ValueType, TENSOR_FLAG_ROW_MAJOR,
// };

use tract_onnx::prelude::tract_data::internal::tract_smallvec::alloc::rc::Rc;
use tract_onnx::prelude::tract_data::internal::tract_smallvec::alloc::sync::Arc;

#[derive(Debug)]
pub struct TractSession {
    pub graph: TractGraph<InferenceFact, Box<dyn InferenceOp>>,
    pub encoding: GraphEncoding,
    pub input_tensors: Option<Vec<Arc<TractTensor>>>,
    // pub input_tensors: Option<TVec<TValue>>,
    // pub output_tensors: Option<SmallVec<[TValue;4]>>,
    pub output_tensors: Option<Vec<Arc<TractTensor>>>,
}

impl TractSession {
    pub fn with_graph(
        graph: TractGraph<InferenceFact, Box<dyn InferenceOp>>,
        encoding: GraphEncoding,
    ) -> Self {
        Self {
            graph,
            encoding,
            input_tensors: None,
            output_tensors: None,
        }
    }
}

#[derive(Default, Clone)]
pub struct TractEngine {
    state: Arc<RwLock<ModelState>>,
}

#[derive(Default)]
pub struct ModelState {
    executions: BTreeMap<GraphExecutionContext, TractSession>,
    models: BTreeMap<Graph, Vec<u8>>,
}

impl ModelState {
    /// Helper function that returns the key that is supposed to be inserted next.
    pub fn key<K: Into<u32> + From<u32> + Copy, V>(&self, keys: Keys<K, V>) -> K {
        match keys.last() {
            Some(&k) => {
                let last: u32 = k.into();
                K::from(last + 1)
            }
            None => K::from(0),
        }
    }
}

#[async_trait]
impl InferenceEngine for TractEngine {
    /// load
    async fn load(&self, model: &[u8]) -> InferenceResult<Graph> {
        let model_bytes = model.to_vec();
        let mut state = self.state.write().await;
        let graph = state.key(state.models.keys());

        log::debug!(
            "load() - inserting graph: {:#?} with size {:#?}",
            graph,
            model_bytes.len()
        );

        state.models.insert(graph, model_bytes);

        log::debug!(
            "load() - current number of models: {:#?}",
            state.models.len()
        );

        Ok(graph)
    }

    /// init_execution_context
    async fn init_execution_context(
        &self,
        graph: Graph,
        target: &ExecutionTarget,
        encoding: &GraphEncoding,
    ) -> InferenceResult<GraphExecutionContext> {
        log::debug!("init_execution_context() - ENTERING");

        log::debug!(
            "init_execution_context() - detected execution target: {:?}",
            target
        );

        log::debug!(
            "init_execution_context() - detected encoding: {:?}",
            encoding
        );

        if !matches!(target, &ExecutionTarget::Cpu) {
            log::error!(
                "This framework does not support execution target '{:?}'",
                target
            );
            return Err(InferenceError::UnsupportedExecutionTarget);
        }

        let mut state = self.state.write().await;
        let mut model_bytes = match state.models.get(&graph) {
            Some(mb) => Cursor::new(mb),
            None => {
                log::error!(
                    "init_execution_context() - cannot find model in state with graph {:#?}",
                    graph
                );
                return Err(InferenceError::RuntimeError);
            }
        };

        let model = match encoding {
            GraphEncoding::Onnx => tract_onnx::onnx()
                .model_for_read(&mut model_bytes)
                .context("failed to get model for read")?,

            GraphEncoding::Tensorflow => tract_tensorflow::tensorflow()
                .model_for_read(&mut model_bytes)
                .context("failed to get model for read")?,

            _ => {
                log::error!(
                    "requested encoding '{:?}' is currently not supported",
                    encoding
                );
                return Err(InferenceError::InvalidEncodingError);
            }
        };

        let gec = state.key(state.executions.keys());

        log::debug!(
            "init_execution_context() - inserting graph execution context: {:#?}",
            gec
        );

        state
            .executions
            .insert(gec, TractSession::with_graph(model, encoding.to_owned()));

        Ok(gec)
    }

    /// set_input
    async fn set_input(
        &self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor,
    ) -> InferenceResult<()> {
        log::debug!(
            "entering set_input() with context: {:?}, index: {}, tensor: {:?}",
            &context,
            index,
            tensor
        );

        let mut state = self.state.write().await;
        let execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "set_input() - cannot find session in state with context {:#?}",
                    context
                );
                return Err(InferenceError::RuntimeError);
            }
        };

        let shape: Vec<usize> = tensor.shape.iter().map(|x| *x as usize).collect();

        execution.graph.set_input_fact(
            index as usize,
            InferenceFact::dt_shape(f32::datum_type(), shape.clone()),
        )?;

        let data: Vec<f32> = bytes_to_f32_vec(tensor.data.as_slice().to_vec()).await?;

        let input: TractTensor = Array::from_shape_vec(shape, data)
            .map_err(|e| InferenceError::ReShapeError(e.to_string()))?
            .into();

        match execution.input_tensors {
            Some(ref mut input_arrays) => {
                // __CB__2022-03-10 re-evaluate next line
                input_arrays.clear();
                input_arrays.push(input.into());

                log::debug!(
                    "set_input() - input arrays now contains {} items",
                    input_arrays.len(),
                );
            }
            None => {
                execution.input_tensors = Some(vec![input.into()]);
            }
        };
        Ok(())
    }

    /// compute()
    async fn compute(&self, context: GraphExecutionContext) -> InferenceResult<()> {
        let mut state = self.state.write().await;
        let execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "compute() - cannot find session in state with context {:#?}",
                    context
                );

                return Err(InferenceError::RuntimeError);
            }
        };

        // TODO
        //
        // There are two `.clone()` calls here that could prove
        // to be *very* ineficient, one in getting the input tensors,
        // the other in making the model runnable.
        // let input_tensors: Vec<Arc<TractTensor>> = execution
        // // let input_tensors = execution
        //     .input_tensors
        //     .as_ref()
        //     .unwrap_or(&vec![])
        //     .clone()
        //     .into_iter()
        //     .collect();

        let input_tensors: SmallVec<[TValue; 4]> = execution
            .input_tensors
            .as_ref()
            .unwrap_or(&vec![])
            .iter() // Use `iter` instead of `into_iter` to avoid consuming the vector
            .map(|arc_tensor| TValue::Const(arc_tensor.clone())) // Wrap each `Arc<Tensor>` in `TValue::Const`
            .collect(); // Collect into `SmallVec<[TValue; 4]>`

        log::debug!(
            "compute() - input tensors contains {} elements",
            input_tensors.len()
        );

        // Some ONNX models don't specify their input tensor
        // shapes completely, so we can only call `.into_optimized()` after we
        // have set the input tensor shapes.
        let output_tensors = execution
            .graph
            .clone()
            .into_optimized()?
            .into_runnable()?
            .run(input_tensors)?;

        log::debug!(
            "compute() - output tensors contains {} elements",
            output_tensors.len()
        );

        // __CB__2022-03-10 re-evaluate next line
        // execution
        //     .output_tensors
        //     .replace(output_tensors.into_iter().collect());

        // Assuming `output_tensors` is a `SmallVec<[TValue; 4]>`
        let output_tensors_vec: Vec<Arc<TractTensor>> = output_tensors
            .into_iter()
            .map(|tvalue| {
                match tvalue {
                    TValue::Const(arc_tensor) => arc_tensor,
                    TValue::Var(rc_tensor) => {
                        // Convert Rc<Tensor> to Arc<Tensor>. This is safe because you own the TValue
                        // and you're not going to use it anymore.
                        Arc::new(Rc::try_unwrap(rc_tensor).expect("Failed to unwrap Rc<Tensor>"))
                    }
                }
            })
            .collect();

        execution.output_tensors.replace(output_tensors_vec);

        Ok(())
    }

    /// get_output
    async fn get_output(
        &self,
        context: GraphExecutionContext,
        index: u32,
    ) -> InferenceResult<Tensor> {
        let state = self.state.read().await;
        let execution = match state.executions.get(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "compute() - cannot find session in state with context {:#?}",
                    context
                );

                return Err(InferenceError::RuntimeError);
            }
        };

        let output_tensors = match execution.output_tensors {
            Some(ref oa) => oa,
            None => {
                log::error!(
                    "get_output() - output_tensors for session is none. 
                    Perhaps you haven't called compute yet?"
                );
                return Err(InferenceError::RuntimeError);
            }
        };

        let tensor = match output_tensors.get(index as usize) {
            //let tensor = match output_tensors.remove(index as usize) {
            Some(a) => a,
            None => {
                log::error!(
                    "get_output() - output_tensors does not contain index {}",
                    index
                );
                return Err(InferenceError::RuntimeError);
            }
        };

        let bytes = f32_array_to_bytes(tensor.as_slice().unwrap()).await;

        let tensor_out = Tensor {
            dtype: DataType::F32,
            shape: tensor
                .shape()
                .iter()
                .cloned()
                .map(|i| i as u32)
                .collect::<Vec<u32>>(),
            data: bytes,
        };

        Ok(tensor_out)
    }

    /// remove model state
    async fn drop_model_state(&self, graph: &Graph, gec: &GraphExecutionContext) {
        let mut state = self.state.write().await;

        state.models.remove(graph);
        state.executions.remove(gec);
    }
}

pub type Result<T> = std::io::Result<T>;

pub async fn bytes_to_f32_vec(data: Vec<u8>) -> Result<Vec<f32>> {
    data.chunks(4)
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_f32::<LittleEndian>()
        })
        .collect()
}

pub async fn f32_array_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut wtr = Vec::with_capacity(values.len() * 4);
    for val in values.iter() {
        // unwrap ok because buf is pre-allocated and won't error
        wtr.write_f32::<LittleEndian>(*val).unwrap();
    }
    wtr
}

// pub async fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
//     let sum: f32 = data.iter().sum();
//     log::debug!(
//         "f32_vec_to_bytes() - flatten output tensor contains {} elements with sum {}",
//         data.len(),
//         sum
//     );
//     let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
//     let result: Vec<u8> = chunks.iter().flatten().copied().collect();

//     log::debug!(
//         "f32_vec_to_bytes() - flatten byte output tensor contains {} elements",
//         result.len()
//     );
//     result
// }