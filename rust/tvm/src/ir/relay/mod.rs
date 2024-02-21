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
use crate::runtime::{self, object::*, IsObjectRef, String as TString};

use super::attrs::Attrs;
use super::expr::BaseExprNode;
use super::function::BaseFuncNode;
use super::span::Span;
use super::ty::Type;

use tvm_macros::Object;
use tvm_rt::NDArray;

pub use super::expr::{GlobalVar, GlobalVarNode};
pub use crate::runtime::DataType;

pub mod attrs;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Expr"]
#[type_key = "RelayExpr"]
pub struct ExprNode {
    pub base: BaseExprNode,
    pub checked_type: Type,
    pub struct_info: ObjectRef,
    pub virtual_device: ObjectRef,
}

impl ExprNode {
    pub fn base<T: IsObject>(span: Span) -> ExprNode {
        ExprNode {
            base: BaseExprNode::base::<T>(span.clone()),
            checked_type: Type::null(),
            struct_info: ObjectRef::null(),
            virtual_device: ObjectRef::null(),
        }
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Id"]
#[type_key = "relay.Id"]
pub struct IdNode {
    pub base: Object,
    pub name_hint: TString,
}

impl Id {
    fn new(name_hint: TString) -> Id {
        let node = IdNode {
            base: Object::base::<IdNode>(),
            name_hint: name_hint,
        };
        Id(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Constant"]
#[type_key = "relay.Constant"]
pub struct ConstantNode {
    pub base: ExprNode,
    pub data: NDArray,
}

impl Constant {
    pub fn new(data: NDArray, span: Span) -> Constant {
        let node = ConstantNode {
            base: ExprNode::base::<ConstantNode>(span),
            data: data,
        };
        Constant(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Tuple"]
#[type_key = "relay.Tuple"]
pub struct TupleNode {
    pub base: ExprNode,
    pub fields: Array<Expr>,
}

impl Tuple {
    pub fn new(fields: Array<Expr>, span: Span) -> Tuple {
        let node = TupleNode {
            base: ExprNode::base::<TupleNode>(span),
            fields,
        };
        Tuple(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Var"]
#[type_key = "relay.Var"]
pub struct VarNode {
    pub base: ExprNode,
    pub vid: Id,
    pub type_annotation: Type,
}

impl Var {
    pub fn new(name_hint: String, type_annotation: Type, span: Span) -> Var {
        let node = VarNode {
            base: ExprNode::base::<VarNode>(span),
            vid: Id::new(name_hint.into()),
            type_annotation: type_annotation,
        };
        Var(Some(ObjectPtr::new(node)))
    }

    pub fn name_hint(&self) -> &TString {
        &self.vid.0.as_ref().unwrap().name_hint
    }

    pub fn static_tensor(name_hint: String, sh: Vec<i32>, dtype: DataType) -> Var {
        let sh = Array::from_vec(sh.into_iter().map(Into::into).collect()).unwrap();
        Self::new(
            name_hint,
            super::ty::TensorType::new(sh, dtype, Span::null()).upcast(),
            Span::null(),
        )
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Call"]
#[type_key = "relay.Call"]
pub struct CallNode {
    pub base: ExprNode,
    deleter: ObjectRef,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: Attrs,
    pub type_args: Array<Type>,
}

impl Call {
    pub fn new(
        op: Expr,
        args: Array<Expr>,
        attrs: Attrs,
        type_args: Array<Type>,
        span: Span,
    ) -> Call {
        let node = CallNode {
            base: ExprNode::base::<CallNode>(span),
            deleter: todo!("Don't know how to construct this"),
            op: op,
            args: args,
            attrs: attrs,
            type_args: type_args,
        };
        Call(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Let"]
#[type_key = "relay.Let"]
pub struct LetNode {
    pub base: ExprNode,
    pub var: Var,
    pub value: Expr,
    pub body: Expr,
}

impl Let {
    pub fn new(var: Var, value: Expr, body: Expr, span: Span) -> Let {
        let node = LetNode {
            base: ExprNode::base::<LetNode>(span),
            var,
            value,
            body,
        };
        Let(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "If"]
#[type_key = "relay.If"]
pub struct IfNode {
    pub base: ExprNode,
    pub cond: Expr,
    pub true_branch: Expr,
    pub false_branch: Expr,
}

impl If {
    pub fn new(cond: Expr, true_branch: Expr, false_branch: Expr, span: Span) -> If {
        let node = IfNode {
            base: ExprNode::base::<IfNode>(span),
            cond,
            true_branch,
            false_branch,
        };
        If(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "TupleGetItem"]
#[type_key = "relay.TupleGetItem"]
pub struct TupleGetItemNode {
    pub base: ExprNode,
    pub tuple: Expr,
    pub index: i32,
}

impl TupleGetItem {
    pub fn new(tuple: Expr, index: i32, span: Span) -> TupleGetItem {
        let node = TupleGetItemNode {
            base: ExprNode::base::<TupleGetItemNode>(span),
            tuple,
            index,
        };
        TupleGetItem(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "RefCreate"]
#[type_key = "relay.RefCreate"]
pub struct RefCreateNode {
    pub base: ExprNode,
    pub value: Expr,
}

impl RefCreate {
    pub fn new(value: Expr, span: Span) -> RefCreate {
        let node = RefCreateNode {
            base: ExprNode::base::<RefCreateNode>(span),
            value,
        };
        RefCreate(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "RefRead"]
#[type_key = "relay.RefRead"]
pub struct RefReadNode {
    pub base: ExprNode,
    pub ref_value: Expr,
}

impl RefRead {
    pub fn new(ref_value: Expr, span: Span) -> RefRead {
        let node = RefReadNode {
            base: ExprNode::base::<RefReadNode>(span),
            ref_value,
        };
        RefRead(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "RefWrite"]
#[type_key = "relay.RefWrite"]
pub struct RefWriteNode {
    pub base: ExprNode,
    pub ref_value: Expr,
    pub value: Expr,
}

impl RefWrite {
    pub fn new(ref_value: Expr, value: Expr, span: Span) -> RefWrite {
        let node = RefWriteNode {
            base: ExprNode::base::<RefWriteNode>(span),
            ref_value,
            value,
        };
        RefWrite(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Constructor"]
#[type_key = "relay.Constructor"]
pub struct ConstructorNode {
    pub base: ExprNode,
    pub name_hint: String,
    pub inputs: Array<Type>,
    pub tag: i32,
}

impl Constructor {
    pub fn new(name_hint: String, inputs: Array<Type>, tag: i32, span: Span) -> Constructor {
        let node = ConstructorNode {
            base: ExprNode::base::<ConstructorNode>(span),
            name_hint,
            inputs,
            tag,
        };
        Constructor(Some(ObjectPtr::new(node)))
    }
}

// TODO(@jroesch): define the type data

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Pattern"]
#[type_key = "relay.Pattern"]
pub struct PatternNode {
    pub base: Object,
    pub span: Span,
}

impl PatternNode {
    pub fn base<T: IsObject>(span: Span) -> PatternNode {
        PatternNode {
            base: Object::base::<T>(),
            span: span,
        }
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PatternWildcard"]
#[type_key = "relay.PatternWildcard"]
pub struct PatternWildcardNode {
    pub base: PatternNode,
}

impl PatternWildcard {
    pub fn new(span: Span) -> PatternWildcard {
        let node = PatternWildcardNode {
            base: PatternNode::base::<PatternWildcardNode>(span),
        };
        PatternWildcard(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PatternVar"]
#[type_key = "relay.PatternVar"]
pub struct PatternVarNode {
    pub base: PatternNode,
    pub var: Var,
}

impl PatternVar {
    pub fn new(var: Var, span: Span) -> PatternVar {
        let node = PatternVarNode {
            base: PatternNode::base::<PatternVarNode>(span),
            var: var,
        };
        PatternVar(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PatternConstructor"]
#[type_key = "relay.PatternConstructor"]
pub struct PatternConstructorNode {
    pub base: PatternNode,
    pub constructor: Constructor,
    pub patterns: Array<Pattern>,
}

impl PatternConstructor {
    pub fn new(
        constructor: Constructor,
        patterns: Array<Pattern>,
        span: Span,
    ) -> PatternConstructor {
        let node = PatternConstructorNode {
            base: PatternNode::base::<PatternConstructorNode>(span),
            constructor,
            patterns,
        };
        PatternConstructor(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PatternTuple"]
#[type_key = "relay.PatternTuple"]
pub struct PatternTupleNode {
    pub base: PatternNode,
    pub patterns: Array<Pattern>,
}

impl PatternTuple {
    pub fn new(patterns: Array<Pattern>, span: Span) -> PatternTuple {
        let node = PatternTupleNode {
            base: PatternNode::base::<PatternTupleNode>(span),
            patterns,
        };
        PatternTuple(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Clause"]
#[type_key = "relay.Clause"]
pub struct ClauseNode {
    pub base: Object,
    pub lhs: Pattern,
    pub rhs: Expr,
}

impl Clause {
    pub fn new(lhs: Pattern, rhs: Expr, _span: Span) -> Clause {
        let node = ClauseNode {
            base: Object::base::<ClauseNode>(),
            lhs,
            rhs,
        };
        Clause(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "Match"]
#[type_key = "relay.Match"]
pub struct MatchNode {
    pub base: ExprNode,
    pub data: Expr,
    pub clauses: Array<Clause>,
    pub complete: bool,
}

impl Match {
    pub fn new(data: Expr, clauses: Array<Clause>, complete: bool, span: Span) -> Match {
        let node = MatchNode {
            base: ExprNode::base::<MatchNode>(span),
            data,
            clauses,
            complete,
        };
        Match(Some(ObjectPtr::new(node)))
    }
}

#[repr(C)]
#[derive(Object, Debug)]
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

    pub fn simple<E>(params: Vec<Var>, body: E) -> Function
    where
        E: IsObjectRef,
        E::Object: AsRef<<Expr as IsObjectRef>::Object>,
    {
        let params = Array::from_vec(params).unwrap();
        Self::new(
            params,
            body.upcast(),
            Type::null(),
            Array::from_vec(vec![]).unwrap(),
        )
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
        let gv = GlobalVar::new("main".to_string(), Span::null());
        let text = as_text(gv.clone());
        assert!(text.contains("@main"));
        Ok(())
    }

    #[test]
    fn test_var() -> Result<()> {
        let var = Var::new("local".to_string(), Type::null(), Span::null());
        let text = as_text(var.clone());
        assert!(text.contains("%local"));
        Ok(())
    }

    #[test]
    fn test_parse_constant() -> Result<()> {
        let module = crate::ir::module::IRModule::parse(
            "",
            r#"
#[version = "0.0.5"]
def @main() -> float32 {
  0.01639530062675476f
}
"#,
        )
        .unwrap();
        let main = module
            .lookup(module.get_global_var("main").unwrap())
            .unwrap();
        let func = main.downcast::<crate::ir::relay::Function>().unwrap();
        let constant = func
            .body
            .clone()
            .downcast::<crate::ir::relay::Constant>()
            .unwrap();
        let tuple_type = constant
            .clone()
            .upcast::<Expr>()
            .checked_type
            .clone()
            .downcast::<crate::ir::ty::TensorType>()
            .unwrap();
        // Test type
        assert_eq!(tuple_type.shape.len(), 0,);
        assert_eq!(tuple_type.dtype, "float32".parse().unwrap(),);
        // Check that actual data matches up with type
        assert_eq!(constant.data.dtype(), "float32".parse().unwrap(),);
        assert_eq!(constant.data.len(), 1);
        assert_eq!(constant.data.size(), 4);
        assert_eq!(constant.data.shape(), &[]);
        Ok(())
    }
}
