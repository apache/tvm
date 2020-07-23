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
    cell::RefCell,
    collections::HashMap,
    ffi::CStr,
    os::raw::{c_char, c_int, c_void},
    pin::Pin,
};

use tvm_sys::{ffi::BackendPackedCFunc, packed_func::PackedFunc};

use crate::{
    threading::{TVMBackendParallelBarrier, TVMBackendParallelLaunch},
    workspace::{TVMBackendAllocWorkspace, TVMBackendFreeWorkspace},
    TVMAPISetLastError,
};

use super::Module;

const TVM_MAIN: &[u8] = b"__tvm_main__";
const TVM_MODULE_CTX: &[u8] = b"__tvm_module_ctx";

/// A module backed by a Dynamic Shared Object (dylib).
pub struct DsoModule<'a> {
    lib: libloading::Library,
    packed_funcs: RefCell<HashMap<String, &'a (dyn PackedFunc)>>,
    _pin: std::marker::PhantomPinned,
}

macro_rules! init_context_func {
    ($lib:ident, $( ($fn:ident, $sig:ty) ),+ $(,)?) => {
        unsafe {
            $(
                let fn_ptr = $lib.get::<*mut $sig>(concat!("__", stringify!($fn)).as_bytes());
                if let Ok(fn_ptr) = fn_ptr {
                    **fn_ptr = $fn;
                }
            )+
        }
    };
}

impl<'a> DsoModule<'a> {
    pub fn new<P: AsRef<std::ffi::OsStr>>(filename: P) -> Result<Pin<Box<Self>>, std::io::Error> {
        let lib = libloading::Library::new(filename)?;

        init_context_func!(
            lib,
            (TVMAPISetLastError, unsafe extern "C" fn(*const i8)),
            (
                TVMBackendAllocWorkspace,
                unsafe extern "C" fn(c_int, c_int, u64, c_int, c_int) -> *mut c_void
            ),
            (
                TVMBackendFreeWorkspace,
                unsafe extern "C" fn(c_int, c_int, *mut c_void) -> c_int
            ),
            (
                TVMBackendParallelLaunch,
                unsafe extern "C" fn(
                    crate::threading::FTVMParallelLambda,
                    *const c_void,
                    usize,
                ) -> c_int
            ),
            (
                TVMBackendParallelBarrier,
                unsafe extern "C" fn(usize, *const tvm_sys::ffi::TVMParallelGroupEnv)
            ),
        );

        // Pin the module in memory so that `ctx` pointer (below) is stable.
        let dso_mod = Box::pin(Self {
            lib,
            packed_funcs: RefCell::new(HashMap::new()),
            _pin: std::marker::PhantomPinned,
        });

        unsafe {
            if let Ok(ctx) = dso_mod.lib.get::<*mut *const c_void>(TVM_MODULE_CTX) {
                **ctx = &dso_mod as *const _ as *const c_void;
            }
        }

        Ok(dso_mod)
    }
}

impl<'a> Module for DsoModule<'a> {
    fn get_function<S: AsRef<str>>(&self, name: S) -> Option<&(dyn PackedFunc)> {
        let name = name.as_ref();
        let func = match unsafe {
            self.lib
                .get::<BackendPackedCFunc>(if name.as_bytes() == TVM_MAIN {
                    // If __tvm_main__ is present, it contains the name of the
                    // actual main function.
                    match self
                        .lib
                        .get::<*const c_char>(TVM_MAIN)
                        .map(|p| CStr::from_ptr(*p))
                    {
                        Ok(m) => m.to_bytes(),
                        _ => return None,
                    }
                } else {
                    name.as_bytes()
                })
        } {
            Ok(func) => unsafe { func.into_raw() },
            Err(_) => return None,
        };

        self.packed_funcs.borrow_mut().insert(
            name.to_string(),
            &*Box::leak(super::wrap_backend_packed_func(name.to_string(), *func)),
        );

        self.packed_funcs.borrow().get(name).copied()
    }
}

impl<'a> Drop for DsoModule<'a> {
    fn drop(&mut self) {
        self.packed_funcs
            .replace(HashMap::new())
            .into_iter()
            .map(|(_name, f)| unsafe { Box::from_raw(f as *const _ as *mut (dyn PackedFunc)) })
            .for_each(std::mem::drop);
    }
}
