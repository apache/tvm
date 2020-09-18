use super::relay;
use crate::runtime::String as TString;
use crate::runtime::{self, external, IsObject, IsObjectRef, Object, ObjectPtr, ObjectRef};
use crate::DataType;

use tvm_macros::Object;

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseExpr"]
#[type_key = "Expr"]
pub struct BaseExprNode {
    pub base: Object,
}

impl BaseExprNode {
    pub fn base<T: IsObject>() -> BaseExprNode {
        BaseExprNode {
            base: Object::base_object::<T>(),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "PrimExpr"]
#[type_key = "PrimExpr"]
pub struct PrimExprNode {
    pub base: BaseExprNode,
    pub datatype: DataType,
}

impl PrimExprNode {
    pub fn base<T: IsObject>(datatype: DataType) -> PrimExprNode {
        PrimExprNode {
            base: BaseExprNode::base::<T>(),
            datatype,
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "GlobalVar"]
#[type_key = "GlobalVar"]
pub struct GlobalVarNode {
    pub base: relay::ExprNode,
    pub name_hint: TString,
}

impl GlobalVar {
    pub fn new(name_hint: String, _span: ObjectRef) -> GlobalVar {
        let node = GlobalVarNode {
            base: relay::ExprNode::base::<GlobalVarNode>(),
            name_hint: name_hint.into(),
        };
        GlobalVar(Some(ObjectPtr::new(node)))
    }
}

// TODO(@jroesch): update to match TVM
// Move IntImm
// Define FloatImm
// Define Bool
// Define tvm::Integer?
// Define RangeNode

// TODO: figure out how to type the last argument runtime::TypedPackedFunc<String(ObjectRef)> annotate)
external! {
    #[name("ir.AsText")]
    fn _as_text(object: ObjectRef, show_meta_data: i32, annotate: runtime::Function) -> TString;
}

pub fn as_text<T: IsObjectRef>(object: T) -> String {
    let no_func = unsafe { runtime::Function::null() };
    _as_text(object.upcast(), 0, no_func)
        .unwrap()
        .as_str()
        .unwrap()
        .into()
}
