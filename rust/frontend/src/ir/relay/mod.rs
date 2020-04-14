
use crate::runtime::{Object, IsObject, ObjectPtr, ObjectRef, String as TString};
use crate::DataType;
use std::str::FromStr;

// macro_rules! define_ref {
//     ($name:ident, $node_type:ident) => {
//         pub struct $name(ObjectPtr<$node_type>);

//         impl $name {
//             fn to_object_ref(self) -> ObjectRef {
//                 ObjectRef(Some(self.0))
//             }
//         }
//     };
// }

#[repr(C)]
pub struct IdNode {
    pub base: Object,
    pub name_hint: TString,
}

unsafe impl IsObject for IdNode {
    const TYPE_KEY: &'static str = "relay.Id";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base
    }
}

#[repr(C)]
pub struct Id(Option<ObjectPtr<IdNode>>);

impl Id {
    fn new(name_hint: TString) -> Id {
        let node = IdNode {
            base: Object::base_object::<IdNode>(),
            name_hint: name_hint,
        };
        Id(Some(ObjectPtr::new(node)))
    }

    fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

// define_ref!(Id, IdNode);

#[repr(C)]
pub struct BaseExpr {
    pub base: Object,
}

#[repr(C)]
pub struct PrimExprNode {
    pub base: BaseExpr,
    pub datatype: DataType,
}

impl BaseExpr {
    fn base<T: IsObject>() -> BaseExpr {
        BaseExpr {
            base: Object::base_object::<T>(),
        }
    }
}

#[repr(C)]
pub struct RelayExpr {
    pub base: BaseExpr,
    pub span: ObjectRef,
    pub checked_type: ObjectRef,
}

impl RelayExpr {
    fn base<T: IsObject>() -> RelayExpr {
        RelayExpr {
            base: BaseExpr::base::<T>(),
            span: ObjectRef::null(),
            checked_type: ObjectRef::null(),
        }
    }
}

#[repr(C)]
pub struct GlobalVarNode {
    pub base: RelayExpr,
    pub name_hint: TString,
}

unsafe impl IsObject for GlobalVarNode {
    const TYPE_KEY: &'static str = "GlobalVar";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base.base.base
    }
}

pub struct GlobalVar(Option<ObjectPtr<GlobalVarNode>>);

impl GlobalVar {
    fn new(name_hint: String, span: ObjectRef) -> GlobalVar {
        let node = GlobalVarNode {
            base: RelayExpr::base::<GlobalVarNode>(),
            // span: span,
            // checked_type: ObjectRef(None),,
            name_hint: TString::new(name_hint).unwrap(),
        };
        GlobalVar(Some(ObjectPtr::new(node)))
    }

    fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

#[repr(C)]
pub struct ConstantNode {
    pub base: RelayExpr,
    pub data: ObjectRef, // make this NDArray.
}

unsafe impl IsObject for VarNode {
    const TYPE_KEY: &'static str = "relay.Var";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base.base.base
    }
}


pub struct Var(Option<ObjectPtr<VarNode>>);

impl Var {
    fn new(name_hint: String, span: ObjectRef) -> Var {
        let node = VarNode {
            base: RelayExpr::base::<VarNode>(),
            vid: Id::new(TString::new(name_hint.to_string()).unwrap()),
            type_annotation: ObjectRef::null(),
        };
        Var(Some(ObjectPtr::new(node)))
    }

    fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

#[repr(C)]
pub struct VarNode {
    pub base: RelayExpr,
    pub vid: Id,
    pub type_annotation: ObjectRef,
}

unsafe impl IsObject for VarNode {
    const TYPE_KEY: &'static str = "relay.Var";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base.base.base
    }
}


pub struct Var(Option<ObjectPtr<VarNode>>);

impl Var {
    fn new(name_hint: String, span: ObjectRef) -> Var {
        let node = VarNode {
            base: RelayExpr::base::<VarNode>(),
            vid: Id::new(TString::new(name_hint.to_string()).unwrap()),
            type_annotation: ObjectRef::null(),
        };
        Var(Some(ObjectPtr::new(node)))
    }

    fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{as_text, String as TString};

    #[test]
    fn test_id() {
        let string = TString::new("foo".to_string()).expect("bar");
        let id = Id::new(string);
        let cstr = as_text(&id.upcast());
        assert!(cstr.into_string().unwrap().contains("relay.Id"));
    }

    #[test]
    fn test_global() {
        let gv = GlobalVar::new("main".to_string(), ObjectRef::null());
        let cstr = as_text(&gv.upcast());
        assert!(cstr.into_string().unwrap().contains("@main"));
    }

    #[test]
    fn test_var() {
        let var = Var::new("local".to_string(), ObjectRef::null());
        let cstr = as_text(&var.upcast());
        assert!(cstr.into_string().unwrap().contains("%local"));
    }
}
