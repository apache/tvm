/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

use crate::runtime::array::Array;
use crate::runtime::{object::*, String as TString};
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
#[type_key = "GlobalVar"]
pub struct GlobalVarNode {
    pub base: RelayExpr,
    pub name_hint: TString,
}

impl GlobalVar {
    pub fn new(name_hint: String, _span: ObjectRef) -> GlobalVar {
        let node = GlobalVarNode {
            base: RelayExpr::base::<GlobalVarNode>(),
            name_hint: name_hint.into(),
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
            vid: Id::new(name_hint.into()),
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
#[ref_name = "BaseFunc"]
#[type_key = "BaseFunc"]
pub struct BaseFuncNode {
    pub base: RelayExpr,
    pub attrs: ObjectRef,
}

impl BaseFuncNode {
    fn base<T: IsObject>() -> BaseFuncNode {
        BaseFuncNode {
            base: RelayExpr::base::<T>(),
            attrs: ObjectRef::null(),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[ref_name = "Function"]
#[type_key = "relay.Function"]
pub struct FunctionNode {
    pub base: BaseFuncNode,
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
            base: BaseFuncNode::base::<FunctionNode>(),
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
    use crate::ir::as_text;
    use crate::runtime::String as TString;
    use anyhow::Result;

    #[test]
    fn test_id() -> Result<()> {
        let string = TString::from("foo");
        let id = Id::new(string);
        let text = as_text(id.clone());
        assert!(text.contains("relay.Id"));
        Ok(())
    }

    #[test]
    fn test_global() -> Result<()> {
        let gv = GlobalVar::new("main".to_string(), ObjectRef::null());
        let text = as_text(gv.clone());
        assert!(text.contains("@main"));
        Ok(())
    }

    #[test]
    fn test_var() -> Result<()> {
        let var = Var::new("local".to_string(), ObjectRef::null());
        let text = as_text(var.clone());
        assert!(text.contains("%local"));
        Ok(())
    }

    use super::Array;
    use crate::ir::relay::Var;
    use crate::runtime::object::ObjectRef;

    #[test]
    fn create_array_and_get() -> Result<()> {
        let vec = vec![
            Var::new("foo".into(), ObjectRef::null()),
            Var::new("bar".into(), ObjectRef::null()),
        ];
        let array = Array::from_vec(vec)?;
        assert_eq!(array.get(0)?.name_hint().to_string(), "foo");
        assert_eq!(array.get(1)?.name_hint().to_string(), "bar");
        Ok(())
    }
}
