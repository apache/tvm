use crate::ir::attrs::BaseAttrsNode;
use crate::ir::PrimExpr;
use crate::runtime::array::Array;
use crate::runtime::DataType;
use crate::runtime::String as TString;
use tvm_macros::Object;

type IndexExpr = PrimExpr;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Conv2DAttrs"]
#[type_key = "relay.attrs.Conv2DAttrs"]
pub struct Conv2DAttrsNode {
    pub base: BaseAttrsNode,
    pub strides: Array<IndexExpr>,
    pub padding: Array<IndexExpr>,
    pub dilation: Array<IndexExpr>,
    // TODO(@gussmith23) groups is "int", what should it be here?
    pub groups: i32,
    pub channels: IndexExpr,
    pub kernel_size: Array<IndexExpr>,
    pub data_layout: TString,
    pub kernel_layout: TString,
    pub out_layout: TString,
    pub out_dtype: DataType,
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "BiasAddAttrs"]
#[type_key = "relay.attrs.BiasAddAttrs"]
pub struct BiasAddAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
}
