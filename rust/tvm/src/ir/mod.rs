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

use crate::runtime::String as TString;
use crate::runtime::{self, external, IsObject, IsObjectRef, Object, ObjectRef};
use crate::DataType;
use tvm_macros::Object;

pub mod arith;
pub mod relay;
pub mod tir;

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

#[repr(C)]
#[derive(Object)]
#[ref_name = "BaseExpr"]
#[type_key = "Expr"]
pub struct BaseExprNode {
    pub base: Object,
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
