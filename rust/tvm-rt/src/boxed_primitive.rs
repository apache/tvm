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

use crate::object::Object;
use tvm_macros::Object;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "BoxBool"]
#[type_key = "runtime.BoxBool"]
pub struct BoxBoolNode {
    base: Object,
    value: bool,
}

impl From<bool> for BoxBool {
    fn from(value: bool) -> Self {
        _box_bool(value as i64).expect("Failed to box boolean for FFI")
    }
}

impl Into<bool> for BoxBool {
    fn into(self) -> bool {
        self.value
    }
}

crate::external! {
    #[name("runtime.BoxBool")]
    fn _box_bool(value: i64) -> BoxBool;

    #[name("runtime.UnBoxBool")]
    fn _unbox_bool(boxed: BoxBool) -> i64;
}
