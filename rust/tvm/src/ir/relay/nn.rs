use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::map::Map;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};
use crate::ir::PrimExpr;
use crate::ir::relay::Expr;
use crate::function::ffi::DLDataType;

external! {
    #[name("relay.op.nn._make.conv2d")]
    pub fn conv2d(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;
}


