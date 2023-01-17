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

use tvm_macros::Object;

use crate::runtime::String as TString;
use crate::runtime::{self, external, IsObject, IsObjectRef, Object, ObjectPtr, ObjectRef};
use crate::DataType;

use super::relay;
use super::span::Span;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BaseExpr"]
#[type_key = "Expr"]
pub struct BaseExprNode {
    pub base: Object,
    pub span: Span,
}

impl BaseExprNode {
    pub fn base<T: IsObject>(span: Span) -> BaseExprNode {
        BaseExprNode {
            base: Object::base::<T>(),
            span,
        }
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PrimExpr"]
#[type_key = "PrimExpr"]
pub struct PrimExprNode {
    pub base: BaseExprNode,
    pub datatype: DataType,
}

impl PrimExprNode {
    pub fn base<T: IsObject>(datatype: DataType, span: Span) -> PrimExprNode {
        PrimExprNode {
            base: BaseExprNode::base::<T>(span),
            datatype,
        }
    }
}

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "GlobalVar"]
#[type_key = "GlobalVar"]
pub struct GlobalVarNode {
    pub base: relay::ExprNode,
    pub name_hint: TString,
}

impl GlobalVar {
    pub fn new(name_hint: String, span: Span) -> GlobalVar {
        let node = GlobalVarNode {
            base: relay::ExprNode::base::<GlobalVarNode>(span),
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
    #[name("relay.ir.AsText")]
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
