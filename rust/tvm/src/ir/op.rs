use crate::ir::relay::ExprNode;
use crate::runtime::array::Array;
use crate::runtime::ObjectRef;
use crate::runtime::String as TString;
use tvm_macros::Object;

type FuncType = ObjectRef;
type AttrFieldInfo = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Op"]
#[type_key = "Op"]
pub struct OpNode {
    pub base: ExprNode,
    pub name: TString,
    pub op_type: FuncType,
    pub description: TString,
    pub arguments: Array<AttrFieldInfo>,
    pub attrs_type_key: TString,
    pub attrs_type_index: u32,
    pub num_inputs: i32,
    pub support_level: i32,
}
