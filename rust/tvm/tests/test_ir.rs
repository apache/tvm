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

use std::convert::TryInto;
use std::str::FromStr;
use tvm::ir::IntImmNode;
use tvm::runtime::String as TString;
use tvm::runtime::{debug_print, Object, ObjectPtr, ObjectRef};
use tvm_rt::{call_packed, DLDataType, Function};
use tvm_sys::TVMRetValue;

#[test]
fn test_new_object() -> anyhow::Result<()> {
    let object = Object::base_object::<Object>();
    let ptr = ObjectPtr::new(object);
    assert_eq!(ptr.count(), 1);
    Ok(())
}

#[test]
fn test_new_string() -> anyhow::Result<()> {
    let string = TString::new("hello world!".to_string())?;
    Ok(())
}

#[test]
fn test_obj_build() -> anyhow::Result<()> {
    let int_imm = Function::get("ir.IntImm").expect("Stable TVM API not found.");

    let dt = DLDataType::from_str("int32").expect("Known datatype doesn't convert.");

    let ret_val: ObjectRef = call_packed!(int_imm, dt, 1337)
        .expect("foo")
        .try_into()
        .unwrap();

    debug_print(&ret_val);

    Ok(())
}
