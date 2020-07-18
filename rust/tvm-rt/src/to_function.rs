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

use std::convert::{TryFrom, TryInto};
use std::{
    os::raw::{c_int, c_void},
    ptr, slice,
};

use super::{function::Result, Function};
use crate::errors::Error;

pub use tvm_sys::{ffi, ArgValue, RetValue};

/// A trait representing whether the function arguments
/// and return type can be assigned to a TVM packed function.
///
/// By splitting the conversion to function into two traits
/// we are able to improve error reporting, by splitting the
/// conversion of inputs and outputs to this trait.
///
/// And the implementation of it to `ToFunction`.
pub trait Typed<I, O> {
    fn args(i: &[ArgValue<'static>]) -> Result<I>;
    fn ret(o: O) -> Result<RetValue>;
}

impl<F, O, E> Typed<(), O> for F
where
    F: Fn() -> O,
    Error: From<E>,
    O: TryInto<RetValue, Error = E>,
{
    fn args(_args: &[ArgValue<'static>]) -> Result<()> {
        debug_assert!(_args.len() == 0);
        Ok(())
    }

    fn ret(o: O) -> Result<RetValue> {
        o.try_into().map_err(|e| e.into())
    }
}

impl<F, A, O, E1, E2> Typed<(A,), O> for F
where
    F: Fn(A) -> O,
    Error: From<E1>,
    Error: From<E2>,
    A: TryFrom<ArgValue<'static>, Error = E1>,
    O: TryInto<RetValue, Error = E2>,
{
    fn args(args: &[ArgValue<'static>]) -> Result<(A,)> {
        debug_assert!(args.len() == 1);
        let a: A = args[0].clone().try_into()?;
        Ok((a,))
    }

    fn ret(o: O) -> Result<RetValue> {
        o.try_into().map_err(|e| e.into())
    }
}

impl<F, A, B, O, E1, E2> Typed<(A, B), O> for F
where
    F: Fn(A, B) -> O,
    Error: From<E1>,
    Error: From<E2>,
    A: TryFrom<ArgValue<'static>, Error = E1>,
    B: TryFrom<ArgValue<'static>, Error = E1>,
    O: TryInto<RetValue, Error = E2>,
{
    fn args(args: &[ArgValue<'static>]) -> Result<(A, B)> {
        debug_assert!(args.len() == 2);
        let a: A = args[0].clone().try_into()?;
        let b: B = args[1].clone().try_into()?;
        Ok((a, b))
    }

    fn ret(o: O) -> Result<RetValue> {
        o.try_into().map_err(|e| e.into())
    }
}

impl<F, A, B, C, O, E1, E2> Typed<(A, B, C), O> for F
where
    F: Fn(A, B, C) -> O,
    Error: From<E1>,
    Error: From<E2>,
    A: TryFrom<ArgValue<'static>, Error = E1>,
    B: TryFrom<ArgValue<'static>, Error = E1>,
    C: TryFrom<ArgValue<'static>, Error = E1>,
    O: TryInto<RetValue, Error = E2>,
{
    fn args(args: &[ArgValue<'static>]) -> Result<(A, B, C)> {
        debug_assert!(args.len() == 3);
        let a: A = args[0].clone().try_into()?;
        let b: B = args[1].clone().try_into()?;
        let c: C = args[2].clone().try_into()?;
        Ok((a, b, c))
    }

    fn ret(o: O) -> Result<RetValue> {
        o.try_into().map_err(|e| e.into())
    }
}

pub trait ToFunction<I, O>: Sized {
    type Handle;

    fn into_raw(self) -> *mut Self::Handle;

    fn call(handle: *mut Self::Handle, args: &[ArgValue<'static>]) -> Result<RetValue>
    where
        Self: Typed<I, O>;

    fn drop(handle: *mut Self::Handle);

    fn to_function(self) -> Function
    where
        Self: Typed<I, O>,
    {
        let mut fhandle = ptr::null_mut() as ffi::TVMFunctionHandle;
        let resource_handle = self.into_raw();

        check_call!(ffi::TVMFuncCreateFromCFunc(
            Some(Self::tvm_callback),
            resource_handle as *mut _,
            None, // Some(Self::tvm_finalizer),
            &mut fhandle as *mut ffi::TVMFunctionHandle,
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
        resource_handle: *mut c_void,
    ) -> c_int
    where
        Self: Typed<I, O>,
    {
        #![allow(unused_assignments, unused_unsafe)]
        // turning off the incorrect linter complaints
        let len = num_args as usize;
        let args_list = slice::from_raw_parts_mut(args, len);
        let type_codes_list = slice::from_raw_parts_mut(type_codes, len);
        let mut local_args: Vec<ArgValue> = Vec::new();
        let mut value = ffi::TVMValue { v_int64: 0 };
        let mut tcode = 0;
        let resource_handle = resource_handle as *mut Self::Handle;
        for i in 0..len {
            value = args_list[i];
            tcode = type_codes_list[i];
            if tcode == ffi::TVMArgTypeCode_kTVMObjectHandle as c_int
                || tcode == ffi::TVMArgTypeCode_kTVMPackedFuncHandle as c_int
                || tcode == ffi::TVMArgTypeCode_kTVMModuleHandle as c_int
            {
                check_call!(ffi::TVMCbArgToReturn(
                    &mut value as *mut _,
                    &mut tcode as *mut _
                ));
            }
            let arg_value = ArgValue::from_tvm_value(value, tcode as u32);
            local_args.push(arg_value);
        }

        let rv = match Self::call(resource_handle, local_args.as_slice()) {
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

impl<O, F> ToFunction<(), O> for F
where
    F: Fn() -> O + 'static,
{
    type Handle = Box<dyn Fn() -> O + 'static>;

    fn into_raw(self) -> *mut Self::Handle {
        let ptr: Box<Self::Handle> = Box::new(Box::new(self));
        Box::into_raw(ptr)
    }

    fn call(handle: *mut Self::Handle, _: &[ArgValue<'static>]) -> Result<RetValue>
    where
        F: Typed<(), O>,
    {
        // Ideally we shouldn't need to clone, probably doesn't really matter.
        let out = unsafe { (*handle)() };
        F::ret(out)
    }

    fn drop(_: *mut Self::Handle) {}
}

macro_rules! to_function_instance {
    ($(($param:ident,$index:tt),)+) => {
        impl<F, $($param,)+ O> ToFunction<($($param,)+), O> for
        F where F: Fn($($param,)+) -> O + 'static {
            type Handle = Box<dyn Fn($($param,)+) -> O + 'static>;

            fn into_raw(self) -> *mut Self::Handle {
                let ptr: Box<Self::Handle> = Box::new(Box::new(self));
                Box::into_raw(ptr)
            }

            fn call(handle: *mut Self::Handle, args: &[ArgValue<'static>]) -> Result<RetValue> where F: Typed<($($param,)+), O> {
                // Ideally we shouldn't need to clone, probably doesn't really matter.
                let args = F::args(args)?;
                let out = unsafe {
                    (*handle)($(args.$index),+)
                };
                F::ret(out)
            }

            fn drop(_: *mut Self::Handle) {}
        }
    }
}

to_function_instance!((A, 0),);
to_function_instance!((A, 0), (B, 1),);
to_function_instance!((A, 0), (B, 1), (C, 2),);
to_function_instance!((A, 0), (B, 1), (C, 2), (D, 3),);

#[cfg(test)]
mod tests {
    use super::{Function, ToFunction, Typed};

    fn zero() -> i32 {
        10
    }

    fn helper<F, I, O>(f: F) -> Function
    where
        F: ToFunction<I, O>,
        F: Typed<I, O>,
    {
        f.to_function()
    }

    #[test]
    fn test_to_function0() {
        helper(zero);
    }

    fn one_arg(i: i32) -> i32 {
        i
    }

    #[test]
    fn test_to_function1() {
        helper(one_arg);
    }

    fn two_arg(i: i32, j: i32) -> i32 {
        i + j
    }

    #[test]
    fn test_to_function2() {
        helper(two_arg);
    }
}
