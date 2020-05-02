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

//! This module provides an idiomatic Rust API for creating and working with TVM functions.
//!
//! For calling an already registered TVM function use [`function::Builder`]
//! To register a TVM packed function from Rust side either
//! use [`function::register`] or the macro [`register_global_func`].
//!
//! See the tests and examples repository for more examples.

use std::{
    mem::{MaybeUninit},
    os::raw::{c_int, c_void},
    ptr, slice,
};

use anyhow::{Result};

pub use tvm_sys::{ffi, ArgValue, RetValue};

use super::Function;

pub trait ToFunction<I, O>: Sized {
    type Handle;

    fn into_raw(self) -> *mut Self::Handle;
    fn call(handle: *mut Self::Handle, args: &[ArgValue]) -> anyhow::Result<RetValue>;
    fn drop(handle: *mut Self::Handle);

    fn to_function(self) -> Function {
        let mut fhandle = ptr::null_mut() as ffi::TVMFunctionHandle;
        let resource_handle = self.into_raw();
        check_call!(ffi::TVMFuncCreateFromCFunc(
            Some(Self::tvm_callback),
            resource_handle as *mut _,
            Some(Self::tvm_finalizer),
            &mut fhandle as *mut _
        ));
        Function::new(fhandle)
    }

    /// The callback function which is wrapped converted by TVM
    /// into a packed function stored in fhandle.
    unsafe extern "C" fn tvm_callback(
        args: *mut ffi::TVMValue,
        type_codes: *mut c_int,
        num_args: c_int,
        ret: ffi::TVMRetValueHandle,
        fhandle: *mut c_void,
    ) -> c_int {
        // turning off the incorrect linter complaints
        #![allow(unused_assignments, unused_unsafe)]
        let len = num_args as usize;
        let args_list = slice::from_raw_parts_mut(args, len);
        let type_codes_list = slice::from_raw_parts_mut(type_codes, len);
        let mut local_args: Vec<ArgValue> = Vec::new();
        let mut value = MaybeUninit::uninit().assume_init();
        let mut tcode = MaybeUninit::uninit().assume_init();
        let rust_fn = fhandle as *mut Self::Handle;
        for i in 0..len {
            value = args_list[i];
            println!("{:?}", value.v_handle);
            tcode = type_codes_list[i];
            if tcode == ffi::TVMTypeCode_kTVMObjectHandle as c_int
                || tcode == ffi::TVMTypeCode_kTVMPackedFuncHandle as c_int
                || tcode == ffi::TVMTypeCode_kTVMModuleHandle as c_int
            {
                check_call!(ffi::TVMCbArgToReturn(
                    &mut value as *mut _,
                    &mut tcode as *mut _
                ));
                println!("{:?}", value.v_handle);
            }
            let arg_value = ArgValue::from_tvm_value(value, tcode as u32);
            println!("{:?}", arg_value);
            local_args.push(arg_value);
        }

        let rv = match Self::call(rust_fn, local_args.as_slice()) {
            Ok(v) => v,
            Err(msg) => {
                crate::set_last_error(&msg);
                return -1;
            }
        };

        let (mut ret_val, ret_tcode) = rv.to_tvm_value();
        let mut ret_type_code = ret_tcode as c_int;
        check_call!(ffi::TVMCFuncSetReturn(
            ret,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _,
            1 as c_int
        ));
        0
    }

    /// The finalizer which is invoked when the packed function's
    /// reference count is zero.
    unsafe extern "C" fn tvm_finalizer(fhandle: *mut c_void) {
        let handle = std::mem::transmute(fhandle);
        Self::drop(handle)
    }
}

impl<'a, 'b> ToFunction<&'a [ArgValue<'b>], RetValue> for fn(&[ArgValue]) -> Result<RetValue> {
    type Handle = for <'x, 'y> fn(&'x [ArgValue<'y>]) -> Result<RetValue>;

    fn into_raw(self) -> *mut Self::Handle {
        self as *mut Self::Handle
    }

    fn call(handle: *mut Self::Handle, args: &[ArgValue]) -> Result<RetValue> {
        unsafe { (*handle)(args) }
    }

    // Function's don't need de-allocation because the pointers are into the code section of memory.
    fn drop(_: *mut Self::Handle) {}
}

impl<'a, O: Into<RetValue>, F> ToFunction<(), O> for F where F: Fn() -> O + 'static {
    type Handle = Box<dyn Fn() -> O + 'static>;

    fn into_raw(self) -> *mut Self::Handle {
        let ptr: Box<Self::Handle> = Box::new(Box::new(self));
        Box::into_raw(ptr)
    }

    fn call(handle: *mut Self::Handle, _: &[ArgValue]) -> Result<RetValue> {
        // Ideally we shouldn't need to clone, probably doesn't really matter.
        unsafe { Ok((*handle)().into()) }
    }

    fn drop(_: *mut Self::Handle) {}
}

macro_rules! to_function_instance {
    ($(($param:ident,$index:expr),)+) => {
        impl<'a, $($param,)+ O: Into<RetValue>, F> ToFunction<($($param,)+), O> for
        F where F: Fn($($param,)+) -> O + 'static,
                $($param: for<'x> From<ArgValue<'x>>,)+  {
            type Handle = Box<dyn Fn($($param,)+) -> O + 'static>;

            fn into_raw(self) -> *mut Self::Handle {
                let ptr: Box<Self::Handle> = Box::new(Box::new(self));
                Box::into_raw(ptr)
            }

            fn call(handle: *mut Self::Handle, args: &[ArgValue]) -> Result<RetValue> {
                // Ideally we shouldn't need to clone, probably doesn't really matter.
                let res = unsafe {
                    (*handle)($(args[$index].clone().into(),)+)
                };
                Ok(res.into())
            }

            fn drop(_: *mut Self::Handle) {}
        }
    }
}

to_function_instance!((A, 0),);
to_function_instance!((A, 0), (B, 1),);
to_function_instance!((A, 0), (B, 1), (C, 2),);
to_function_instance!((A, 0), (B, 1), (C, 2), (D, 3),);
