use crate::runtime::Object;
use tvm_macros::Object;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Attrs"]
#[type_key = "Attrs"]
pub struct BaseAttrsNode {
    pub base: Object,
}
