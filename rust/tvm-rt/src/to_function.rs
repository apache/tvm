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
    mem::MaybeUninit,
    os::raw::{c_int, c_void},
    ptr, slice,
};

use anyhow::Result;

pub use tvm_sys::{ffi, ArgValue, RetValue};

use super::Function;
use std::convert::{TryFrom, TryInto};

/// A trait representing whether the function arguments
/// and return type can be assigned to a TVM packed function.
///
/// By splitting the conversion to function into two traits
/// we are able to improve error reporting, by splitting the
/// conversion of inputs and outputs to this trait.
///
/// And the implementation of it to `ToFunction`.
pub trait Typed<I, O> {
    fn args(i: &[ArgValue<'static>]) -> anyhow::Result<I>;
    fn ret(o: O) -> RetValue;
}

impl<'a, F> Typed<&'a [ArgValue<'static>], anyhow::Result<RetValue>> for F
where
    F: Fn(&'a [ArgValue]) -> anyhow::Result<RetValue>,
{
    fn args(args: &[ArgValue<'static>]) -> anyhow::Result<&'a [ArgValue<'static>]> {
        // this is BAD but just hacking for time being
        Ok(unsafe { std::mem::transmute(args) })
    }

    fn ret(ret_value: anyhow::Result<RetValue>) -> RetValue {
        ret_value.unwrap()
    }
}

impl<F, O: Into<RetValue>> Typed<(), O> for F
where
    F: Fn() -> O,
{
    fn args(_args: &[ArgValue<'static>]) -> anyhow::Result<()> {
        debug_assert!(_args.len() == 0);
        Ok(())
    }

    fn ret(o: O) -> RetValue {
        o.into()
    }
}

impl<F, A, O: Into<RetValue>, E: Into<anyhow::Error>> Typed<(A,), O> for F
where
    F: Fn(A) -> O,
    E: std::error::Error + Send + Sync + 'static,
    A: TryFrom<ArgValue<'static>, Error = E>,
{
    fn args(args: &[ArgValue<'static>]) -> anyhow::Result<(A,)> {
        debug_assert!(args.len() == 1);
        let a: A = args[0].clone().try_into()?;
        Ok((a,))
    }

    fn ret(o: O) -> RetValue {
        o.into()
    }
}

impl<F, A, B, O: Into<RetValue>, E: Into<anyhow::Error>> Typed<(A, B), O> for F
where
    F: Fn(A, B) -> O,
    E: std::error::Error + Send + Sync + 'static,
    A: TryFrom<ArgValue<'static>, Error = E>,
    B: TryFrom<ArgValue<'static>, Error = E>,
{
    fn args(args: &[ArgValue<'static>]) -> anyhow::Result<(A, B)> {
        debug_assert!(args.len() == 2);
        let a: A = args[0].clone().try_into()?;
        let b: B = args[1].clone().try_into()?;
        Ok((a, b))
    }

    fn ret(o: O) -> RetValue {
        o.into()
    }
}

impl<F, A, B, C, O: Into<RetValue>, E: Into<anyhow::Error>> Typed<(A, B, C), O> for F
where
    F: Fn(A, B, C) -> O,
    E: std::error::Error + Send + Sync + 'static,
    A: TryFrom<ArgValue<'static>, Error = E>,
    B: TryFrom<ArgValue<'static>, Error = E>,
    C: TryFrom<ArgValue<'static>, Error = E>,
{
    fn args(args: &[ArgValue<'static>]) -> anyhow::Result<(A, B, C)> {
        debug_assert!(args.len() == 3);
        let a: A = args[0].clone().try_into()?;
        let b: B = args[1].clone().try_into()?;
        let c: C = args[2].clone().try_into()?;
        Ok((a, b, c))
    }

    fn ret(o: O) -> RetValue {
        o.into()
    }
}

pub trait ToFunction<I, O>: Sized {
    type Handle;

    fn into_raw(self) -> *mut Self::Handle;

    fn call(handle: *mut Self::Handle, args: &[ArgValue<'static>]) -> anyhow::Result<RetValue>
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
    ) -> c_int
    where
        Self: Typed<I, O>,
    {
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

// /// A wrapper that is used to work around inference issues for bare functions.
// ///
// /// Used to implement `register_untyped`.
// pub(self) struct RawFunction {
//     fn_ptr: for<'a> fn (&'a [ArgValue<'static>]) -> Result<RetValue>
// }

// impl RawFunction {
//     fn new(fn_ptr: for<'a> fn (&'a [ArgValue<'static>]) -> Result<RetValue>) -> RawFunction {
//         RawFunction { fn_ptr: fn_ptr }
//     }
// }

// impl Typed<&[ArgValue<'static>], ()> for RawFunction {
//     fn args(i: &[ArgValue<'static>]) -> anyhow::Result<&[ArgValue<'static>]> {
//         Ok(i)
//     }

//     fn ret(o: O) -> RetValue;
// }

// impl ToFunction<(), ()> for RawFunction
// {
//     type Handle = fn(&[ArgValue<'static>]) -> Result<RetValue>;

//     fn into_raw(self) -> *mut Self::Handle {
//         self.fn_ptr as *mut Self::Handle
//     }

//     fn call(handle: *mut Self::Handle, args: &[ArgValue<'static>]) -> Result<RetValue> {
//         let handle: Self::Handle = unsafe { std::mem::transmute(handle) };
//         let r = handle(args);
//         println!("afters");
//         r
//     }

//     // Function's don't need de-allocation because the pointers are into the code section of memory.
//     fn drop(_: *mut Self::Handle) {}
// }

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
        Ok(F::ret(out))
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
                Ok(F::ret(out))
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
    // use super::RawFunction;
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

    // fn func_args(args: &[ArgValue<'static>]) -> anyhow::Result<RetValue> {
    //     Ok(10.into())
    // }

    // #[test]
    // fn test_fn_ptr() {
    //     let raw_fn = RawFunction::new(func_args);
    //     raw_fn.to_function();
    // }

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
