use crate::ir::attrs::BaseAttrsNode;
use tvm_macros::Object;

#[repr(C)]
#[derive(Object)]
#[ref_name = "ExpandDimsAttrs"]
#[type_key = "relay.attrs.ExpandDimsAttrs"]
pub struct ExpandDimsAttrsNode {
    pub base: BaseAttrsNode,
    pub axis: i32,
    pub num_newaxis: i32,
}
