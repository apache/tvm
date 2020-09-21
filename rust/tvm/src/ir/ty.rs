use tvm_macros::Object;
use tvm_rt::{array::Array, DataType};
use crate::runtime::{ObjectRef, Object};

use super::PrimExpr;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Type"]
#[type_key = "Type"]
pub struct TypeNode {
    pub base: Object,
    pub span: ObjectRef,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseTensorType"]
#[type_key = "relay.BaseTensorType"]
pub struct BaseTensorTypeNode {
    pub base: TypeNode,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "TensorType"]
#[type_key = "relay.TensorType"]
pub struct TensorTypeNode {
    pub base: TypeNode,
    pub shape: Array<PrimExpr>,
    pub dtype: DataType,
}
