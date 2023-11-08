use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};
use crate::ir::PrimExpr;
use crate::ir::tir::IntImm;
use crate::ir::relay::Expr;
use crate::function::ffi::DLDataType;

external! {
    #[name("relay.op._make.squeeze")]
    pub fn squeeze(data: Expr, axis: Array<PrimExpr>) -> Expr;

    #[name("relay.op._make.take")]
    pub fn take(data: Expr, indices: Expr, batch_dims: IntImm, axis: IntImm, mode: TVMString) -> Expr;

    #[name("relay.op._make.split")]
    pub fn split(data: Expr, indices_or_sections: ObjectRef, axis: i32) -> Expr;

    #[name("relay.op._make.strided_slice")]
    pub fn strided_slice(data: Expr, begin: Array<PrimExpr>, end: Array<PrimExpr>, strides: Array<PrimExpr>, slice_mode: TVMString, axes: Array<PrimExpr>) -> Expr;
}
