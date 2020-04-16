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

use std::{
    collections::HashMap, convert::AsRef, ffi::CStr, os::raw::c_char, string::String, sync::Mutex,
};

use tvm_common::{ffi::BackendPackedCFunc, packed_func::PackedFunc};

use super::Module;

pub struct SystemLibModule;

#[cfg(target_env = "sgx")]
extern "C" {
    fn __tvm_module_startup();
}

lazy_static! {
    static ref SYSTEM_LIB_FUNCTIONS: Mutex<HashMap<String, &'static (dyn PackedFunc)>> =
        Mutex::new(HashMap::new());
}

impl Module for SystemLibModule {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)> {
        SYSTEM_LIB_FUNCTIONS
            .lock()
            .unwrap()
            .get(name.as_ref())
            .copied()
    }
}

impl Default for SystemLibModule {
    fn default() -> Self {
        #[cfg(target_env = "sgx")]
        unsafe {
            __tvm_module_startup();
        }
        SystemLibModule {}
    }
}

#[no_mangle]
pub extern "C" fn TVMBackendRegisterSystemLibSymbol(
    cname: *const c_char,
    func: BackendPackedCFunc,
) -> i32 {
    let name = unsafe { CStr::from_ptr(cname).to_str().unwrap() };
    SYSTEM_LIB_FUNCTIONS.lock().unwrap().insert(
        name.to_string(),
        &*Box::leak(super::wrap_backend_packed_func(name.to_string(), func)),
    );
    0
}
