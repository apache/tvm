use crate::runtime::{IsObjectRef, IsObject, ObjectRef};
use crate::ir::relay::ExprNode;

use tvm_macros::Object;

// Define Calling Convention.

// TODO(@jroesch): define DictAttrs
pub type DictAttrs = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseFunc"]
#[type_key = "BaseFunc"]
pub struct BaseFuncNode {
    pub base: ExprNode,
    pub attrs: DictAttrs,
}

impl BaseFuncNode {
    pub fn base<T: IsObject>() -> BaseFuncNode {
        BaseFuncNode {
            base: ExprNode::base::<T>(),
            attrs: <ObjectRef as IsObjectRef>::null(),
        }
    }
}
