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

use crate::ir::relay::ExprNode;
use crate::runtime::array::Array;
use crate::runtime::ObjectRef;
use crate::runtime::String as TString;
use tvm_macros::Object;

type FuncType = ObjectRef;
type AttrFieldInfo = ObjectRef;

#[repr(C)]
#[derive(Object, Debug)]
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
