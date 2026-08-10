use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, ErrorCode, Result};

pub const MAX_HTTP_BODY_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_TENSOR_ELEMENTS: usize = 1_000_000;
pub const MAX_INPUTS: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServerMetadata {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub versions: Vec<String>,
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    pub data: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RequestedOutput {
    pub name: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct InferRequest {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
    pub inputs: Vec<InferTensor>,
    #[serde(default)]
    pub outputs: Vec<RequestedOutput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferResponse {
    pub model_name: String,
    pub model_version: String,
    pub id: String,
    pub outputs: Vec<InferTensor>,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ShapeSignature(pub Vec<(String, String, Vec<i64>)>);

impl InferRequest {
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "at least one input is required",
            ));
        }
        if self.inputs.len() > MAX_INPUTS {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                format!("input count exceeds {MAX_INPUTS}"),
            ));
        }
        for input in &self.inputs {
            if input.name.is_empty() {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    "input name must not be empty",
                ));
            }
            if input.shape.is_empty() || input.shape.iter().any(|dimension| *dimension <= 0) {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!("input {} has invalid shape", input.name),
                ));
            }
            let elements = input.shape.iter().try_fold(1_usize, |total, dimension| {
                let dimension = usize::try_from(*dimension).ok()?;
                total.checked_mul(dimension)
            });
            if elements.is_none() || elements.unwrap_or(usize::MAX) > MAX_TENSOR_ELEMENTS {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!(
                        "input {} exceeds {MAX_TENSOR_ELEMENTS} elements",
                        input.name
                    ),
                ));
            }
            if !matches!(input.datatype.as_str(), "FP32" | "FP64" | "INT32" | "INT64") {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!("unsupported datatype {}", input.datatype),
                ));
            }
            if !input.data.is_array() {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!("input {} data must be a JSON array", input.name),
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn shape_signature(&self) -> ShapeSignature {
        let mut inputs = self
            .inputs
            .iter()
            .map(|input| {
                let non_batch = input.shape.get(1..).unwrap_or_default().to_vec();
                (input.name.clone(), input.datatype.clone(), non_batch)
            })
            .collect::<Vec<_>>();
        inputs.sort_unstable();
        ShapeSignature(inputs)
    }

    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.inputs
            .first()
            .and_then(|input| input.shape.first())
            .and_then(|dimension| usize::try_from(*dimension).ok())
            .unwrap_or(1)
    }

    #[must_use]
    pub fn timeout_us(&self) -> Option<u64> {
        self.parameters.get("timeout_us").and_then(Value::as_u64)
    }

    #[must_use]
    pub fn prefix_tokens(&self) -> Option<Vec<u32>> {
        self.parameters
            .get("prefix_tokens")
            .and_then(Value::as_array)
            .map(|tokens| {
                tokens
                    .iter()
                    .filter_map(Value::as_u64)
                    .filter_map(|token| u32::try_from(token).ok())
                    .collect()
            })
    }
}
