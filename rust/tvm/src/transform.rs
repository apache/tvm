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

use crate::ir::relay::Function;
use crate::runtime::array::Array;
use crate::runtime::{
    external,
    function::{self, Result, ToFunction},
    String as TString,
};
use crate::runtime::{Object, ObjectPtr, ObjectRef};

use tvm_macros::Object;

pub type Pass = ObjectRef;
pub type IRModule = ObjectRef;
pub type PassContext = ObjectRef;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "PassInfo"]
#[type_key = "transform.PassInfo"]
pub struct PassInfoNode {
    pub base: Object,
    pub opt_level: i32,
    pub name: TString,
    pub required: Array<TString>,
}

impl PassInfo {
    pub fn new(opt_level: i32, name: String, required: Vec<String>) -> Result<PassInfo> {
        let required = required.into_iter().map(|name| name.into()).collect();

        let required = Array::from_vec(required)?;

        let node = PassInfoNode {
            base: Object::base::<PassInfoNode>(),
            opt_level,
            name: name.into(),
            required,
        };

        Ok(PassInfo(Some(ObjectPtr::new(node))))
    }
}

external! {
    #[name("relay._transform.MakeFunctionPass")]
    fn create_func_pass(func: function::Function, pass_info: PassInfo) -> Pass;
}

pub fn function_pass<F: Fn(Function, IRModule, PassContext) -> Function + 'static>(
    pass_fn: F,
    pass_info: PassInfo,
) -> Result<Pass> {
    let func = pass_fn.to_function();
    create_func_pass(func, pass_info)
}

/// A macro for generating the correct TVM symbols for plugin loading.
///
/// The expression passed to the macro will be run when TVM loads the
/// shared library.
///
/// This is useful for calling register to register packed functions
/// to consume via TVM's packed function APIs.
#[macro_export]
macro_rules! initialize {
    ($body:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn initialize(
            args: *mut tvm_sys::ffi::TVMValue,
            type_codes: *mut c_int,
            num_args: c_int,
            ret: tvm_sys::ffi::TVMRetValueHandle,
        ) -> c_int {
            $body
            return 0;
        }
    };
}

#[macro_export]
macro_rules! export_pass {
    ($name:literal,$func:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn initialize(
            args: *mut tvm_sys::ffi::TVMValue,
            type_codes: *mut c_int,
            num_args: c_int,
            ret: tvm_sys::ffi::TVMRetValueHandle,
        ) -> c_int {
            register($func, $name).unwrap();
            return 0;
        }
    };
}
