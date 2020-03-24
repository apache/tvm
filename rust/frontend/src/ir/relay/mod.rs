
use crate::runtime::{Object, ObjectPtr, ObjectRef, String as TString};
use crate::DataType;

macro_rules! define_ref {
    ($name:ident, $node_type:ident) => {
        pub struct $name(ObjectPtr<$node_type>);

        impl $name {
            fn to_object_ref(self) -> ObjectRef {
                ObjectRef(Some(self.0))
            }
        }
    };
}

#[repr(C)]
pub struct IdNode {
    pub name_hint: TString,
}

define_ref!(Id, IdNode);

#[repr(C)]
pub struct BaseExpr {
    pub base: Object,
    pub datatype: DataType,
}

#[repr(C)]
pub struct RelayExpr {
    pub base: BaseExpr,
    pub span: ObjectRef,
    pub checked_type: ObjectRef,
}

#[repr(C)]
pub struct GlobalVarNode {
    pub base: RelayExpr,
    pub name_hint: TString,
}

define_ref!(GlobalVar, GlobalVarNode);

impl GlobalVar {
    fn new(name_hint: String, span: ObjectRef) -> GlobalVar {
        GlobalVar(GlobalVarNode {
            base: RelayExpr {
                base: BaseExpr {
                    base: Object::base_object::<GlobalVarNode>(),
                    datatype: DataType { code: 0, bits: 0, lanes: 0 },
                },
                span: span,
                checked_type: ObjectRef(None),
            },
            name_hint,
        })
    }
}
