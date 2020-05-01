use super::array::Array;
use crate::runtime::{IsObject, Object, ObjectPtr, ObjectRef, String as TString, ToObjectRef};
use crate::DataType;
use tvm_macros::Object;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Id"]
#[type_key = "relay.Id"]
pub struct IdNode {
    pub base: Object,
    pub name_hint: TString,
}

impl Id {
    fn new(name_hint: TString) -> Id {
        let node = IdNode {
            base: Object::base_object::<IdNode>(),
            name_hint: name_hint,
        };
        Id(Some(ObjectPtr::new(node)))
    }
}

// define_ref!(Id, IdNode);

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseExpr"]
#[type_key = "Expr"]
pub struct BaseExprNode {
    pub base: Object,
}

#[repr(C)]
pub struct PrimExprNode {
    pub base: BaseExprNode,
    pub datatype: DataType,
}

impl BaseExprNode {
    fn base<T: IsObject>() -> BaseExprNode {
        BaseExprNode {
            base: Object::base_object::<T>(),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Expr"]
#[type_key = "relay.Expr"]
pub struct RelayExpr {
    pub base: BaseExprNode,
    pub span: ObjectRef,
    pub checked_type: ObjectRef,
}

impl RelayExpr {
    fn base<T: IsObject>() -> RelayExpr {
        RelayExpr {
            base: BaseExprNode::base::<T>(),
            span: ObjectRef::null(),
            checked_type: ObjectRef::null(),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "GlobalVar"]
#[type_key = "relay.GlobalVar"]
pub struct GlobalVarNode {
    pub base: RelayExpr,
    pub name_hint: TString,
}

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
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Constant"]
#[type_key = "relay.Constant"]
pub struct ConstantNode {
    pub base: RelayExpr,
    pub data: ObjectRef, // make this NDArray.
}

impl Constant {
    pub fn new(data: ObjectRef, _span: ObjectRef) -> Constant {
        let node = ConstantNode {
            base: RelayExpr::base::<ConstantNode>(),
            data: data,
        };
        Constant(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Var"]
#[type_key = "relay.Var"]
pub struct VarNode {
    pub base: RelayExpr,
    pub vid: Id,
    pub type_annotation: ObjectRef,
}

impl Var {
    pub fn new(name_hint: String, _span: ObjectRef) -> Var {
        let node = VarNode {
            base: RelayExpr::base::<VarNode>(),
            vid: Id::new(TString::new(name_hint.to_string()).unwrap()),
            type_annotation: ObjectRef::null(),
        };
        Var(Some(ObjectPtr::new(node)))
    }

    pub fn name_hint(&self) -> &TString {
        &self.vid.0.as_ref().unwrap().name_hint
    }

    pub fn to_expr(self) -> Expr {
        unsafe { Expr(std::mem::transmute(self.0)) }
    }
}

pub type Type = ObjectRef;
pub type Attrs = ObjectRef;

#[repr(C)]
#[derive(Object)]
#[ref_name = "Call"]
#[type_key = "relay.Call"]
pub struct CallNode {
    pub base: RelayExpr,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: ObjectRef,
    pub type_args: Array<ObjectRef>,
}

impl Call {
    pub fn new(
        op: Expr,
        args: Array<Expr>,
        attrs: Attrs,
        type_args: Array<ObjectRef>,
        _span: ObjectRef,
    ) -> Call {
        let node = CallNode {
            base: RelayExpr::base::<VarNode>(),
            op: op,
            args: args,
            attrs: attrs,
            type_args: type_args,
        };
        Call(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Function"]
#[type_key = "relay.Function"]
pub struct FunctionNode {
    pub base: RelayExpr,
    pub params: Array<Var>,
    pub body: Expr,
    pub ret_type: Type,
    pub type_params: Array<Type>,
}

impl Function {
    pub fn new(
        params: Array<Var>,
        body: Expr,
        ret_type: Type,
        type_params: Array<Type>,
    ) -> Function {
        let node = FunctionNode {
            base: RelayExpr::base::<FunctionNode>(),
            params: params,
            body: body,
            ret_type: ret_type,
            type_params: type_params,
        };
        Function(Some(ObjectPtr::new(node)))
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
