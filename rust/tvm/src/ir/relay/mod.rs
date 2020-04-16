
use crate::runtime::{Object, IsObject, ObjectPtr, ObjectRef, ToObjectRef, String as TString};
use crate::DataType;
use std::convert::TryFrom;
use tvm_rt::TVMRetValue;

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
    pub fn new(name_hint: String, _span: ObjectRef) -> GlobalVar {
        let node = GlobalVarNode {
            base: RelayExpr::base::<GlobalVarNode>(),
            // span: span,
            // checked_type: ObjectRef(None),,
            name_hint: TString::new(name_hint).unwrap(),
        };
        GlobalVar(Some(ObjectPtr::new(node)))
    }

    pub fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

#[repr(C)]
pub struct ConstantNode {
    pub base: RelayExpr,
    pub data: ObjectRef, // make this NDArray.
}

unsafe impl IsObject for ConstantNode {
    const TYPE_KEY: &'static str = "relay.Constant";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base.base.base
    }
}

pub struct Constant(Option<ObjectPtr<ConstantNode>>);

impl Constant {
    pub fn new(data: ObjectRef, _span: ObjectRef) -> Constant {
        let node = ConstantNode {
            base: RelayExpr::base::<ConstantNode>(),
            data: data,
        };
        Constant(Some(ObjectPtr::new(node)))
    }

    pub fn upcast(&self) -> ObjectRef {
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
    pub fn new(name_hint: String, _span: ObjectRef) -> Var {
        let node = VarNode {
            base: RelayExpr::base::<VarNode>(),
            vid: Id::new(TString::new(name_hint.to_string()).unwrap()),
            type_annotation: ObjectRef::null(),
        };
        Var(Some(ObjectPtr::new(node)))
    }

    pub fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }

    pub fn name_hint(&self) -> &TString {
        &self.vid.0.as_ref().unwrap().name_hint
    }
}

impl ToObjectRef for Var {
    fn to_object_ref(&self) -> ObjectRef {
        self.upcast()
    }
}

impl std::ops::Deref for Var {
    type Target = VarNode;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}


impl TryFrom<TVMRetValue> for Var {
    type Error = anyhow::Error;

    fn try_from(ret_val: TVMRetValue) -> Result<Var, Self::Error> {
        let oref: ObjectRef = ret_val.try_into()?;
        oref.downcast::<Var>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{as_text, String as TString};
    use anyhow::Result;

    #[test]
    fn test_id() -> Result<()> {
        let string = TString::new("foo".to_string()).expect("bar");
        let id = Id::new(string);
        let cstr = as_text(&id.upcast())?;
        assert!(cstr.into_string()?.contains("relay.Id"));
        Ok(())
    }

    #[test]
    fn test_global() -> Result<()> {
        let gv = GlobalVar::new("main".to_string(), ObjectRef::null());
        let cstr = as_text(&gv.upcast())?;
        assert!(cstr.into_string()?.contains("@main"));
        Ok(())
    }

    #[test]
    fn test_var() -> Result<()> {
        let var = Var::new("local".to_string(), ObjectRef::null());
        let cstr = as_text(&var.upcast())?;
        assert!(cstr.into_string()?.contains("%local"));
        Ok(())
    }
}
