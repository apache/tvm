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

use super::span::Span;

use crate::ir::relay::ExprNode;
use crate::runtime::{IsObject, IsObjectRef, ObjectRef};

// TODO(@jroesch): define DictAttrs
pub type DictAttrs = ObjectRef;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BaseFunc"]
#[type_key = "BaseFunc"]
pub struct BaseFuncNode {
    pub base: ExprNode,
    pub attrs: DictAttrs,
}

impl BaseFuncNode {
    pub fn base<T: IsObject>() -> BaseFuncNode {
        BaseFuncNode {
            base: ExprNode::base::<T>(Span::null()),
            attrs: <ObjectRef as IsObjectRef>::null(),
        }
    }
}
