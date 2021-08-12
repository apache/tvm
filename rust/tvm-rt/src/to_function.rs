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
use crate::AsArgValue;
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
pub trait Typed<'a, I, O> {
    fn args(i: Vec<ArgValue<'a>>) -> Result<I>;
    fn ret(o: O) -> Result<RetValue>;
}

trait AsArgValueErased where Self: for<'a> AsArgValue<'a> {
    fn as_arg_value<'a>(&'a self) -> ArgValue<'a>;
}

struct ArgList {
    args: Vec<Box<dyn AsArgValueErased>>
}

impl<T: 'static> AsArgValueErased for T where T: for<'a> AsArgValue<'a> {
    fn as_arg_value<'a>(&'a self) -> ArgValue<'a> {
        AsArgValue::as_arg_value(self)
    }
}

pub trait ToFunction<I, O>: Sized {
    type Handle;

    fn into_raw(self) -> *mut Self::Handle;

    fn call<'a>(handle: *mut Self::Handle, args: Vec<ArgValue<'a>>) -> Result<RetValue>
    where
        Self: for<'arg> Typed<'arg, I, O>;

    fn drop(handle: *mut Self::Handle);

    fn to_function(self) -> Function
    where
        Self: for<'a> Typed<'a, I, O>,
    {
        let mut fhandle = ptr::null_mut() as ffi::TVMFunctionHandle;
        let resource_handle = self.into_raw();

        check_call!(ffi::TVMFuncCreateFromCFunc(
            Some(Self::tvm_callback),
            resource_handle as *mut _,
            None, // Some(Self::tvm_finalizer),
            &mut fhandle as *mut ffi::TVMFunctionHandle,
        ));

        Function::from_raw(fhandle)
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
        Self: for <'a> Typed<'a,  I, O>,
    {
        #![allow(unused_assignments, unused_unsafe)]
        let result = std::panic::catch_unwind(|| {
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
                    || tcode == ffi::TVMArgTypeCode_kTVMObjectRValueRefArg as c_int
                    || tcode == ffi::TVMArgTypeCode_kTVMPackedFuncHandle as c_int
                    || tcode == ffi::TVMArgTypeCode_kTVMModuleHandle as c_int
                    || tcode == ffi::TVMArgTypeCode_kTVMNDArrayHandle as c_int
                {
                    check_call!(ffi::TVMCbArgToReturn(
                        &mut value as *mut _,
                        &mut tcode as *mut _
                    ));
                }
                let arg_value = ArgValue::from_tvm_value(value, tcode as u32);
                local_args.push(arg_value);
            }

            // Ref-count be 2.
            let rv = match Self::call(resource_handle, local_args) {
                Ok(v) => v,
                Err(msg) => {
                    return Err(msg);
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

            Ok(())
        });

        // Here we handle either a panic or true error to isolate
        // the unwinding as it will cause issues if we allow Rust
        // to unwind over C++ boundary without care.
        match result {
            Err(_) => {
                // TODO(@jroesch): figure out how to improve error here.
                crate::set_last_error(&Error::Panic);
                return -1;
            }
            Ok(inner_res) => match inner_res {
                Err(err) => {
                    crate::set_last_error(&err);
                    return -1;
                }
                Ok(()) => return 0,
            },
        }
    }

    /// The finalizer which is invoked when the packed function's
    /// reference count is zero.
    unsafe extern "C" fn tvm_finalizer(fhandle: *mut c_void) {
        let handle = std::mem::transmute(fhandle);
        Self::drop(handle)
    }
}

// impl<'a> Typed<'a, Vec<ArgValue<'a>>, RetValue> for for<'arg> fn(Vec<ArgValue<'arg>>) -> Result<RetValue> {
//     fn args(args: Vec<ArgValue<'a>>) -> Result<Vec<ArgValue<'a>>> {
//         Ok(args)
//     }

//     fn ret(o: RetValue) -> Result<RetValue> {
//         Ok(o)
//     }
// }

// impl ToFunction<ArgList, RetValue>
//     for fn(ArgList) -> Result<RetValue>
// {
//     type Handle = for<'a> fn(Vec<ArgValue<'a>>) -> Result<RetValue>;

//     fn into_raw(self) -> *mut Self::Handle {
//         let ptr: Box<Self::Handle> = Box::new(self);
//         Box::into_raw(ptr)
//     }

//     fn call<'a>(handle: *mut Self::Handle, args: Vec<ArgValue<'a>>) -> Result<RetValue> {
//         unsafe {
//             let func = (*handle);
//             func(args)
//         }
//     }

//     fn drop(_: *mut Self::Handle) {}
// }

macro_rules! impl_typed_and_to_function {
    ($len:literal; $($t:ident),*) => {
        impl<'a, F, Out, $($t),*> Typed<'a, ($($t,)*), Out> for F
        where
            F: Fn($($t),*) -> Out,
            Out: TryInto<RetValue>,
            Error: From<Out::Error>,
            $( $t: TryFrom<ArgValue<'a>>,
               Error: From<<$t as TryFrom<ArgValue<'a>>>::Error>, )*
        {
            #[allow(non_snake_case, unused_variables, unused_mut)]
            fn args(args: Vec<ArgValue<'a>>) -> Result<($($t,)*)> {
                if args.len() != $len {
                    return Err(Error::CallFailed(format!("{} expected {} arguments, got {}.\n",
                                                         std::any::type_name::<Self>(),
                                                         $len, args.len())))
                }
                let mut args = args.into_iter();
                $(let $t = args.next().unwrap().try_into()?;)*
                Ok(($($t,)*))
            }

            fn ret(out: Out) -> Result<RetValue> {
                out.try_into().map_err(|e| e.into())
            }
        }


        impl<F, $($t,)* Out> ToFunction<($($t,)*), Out> for F
        where
            F: Fn($($t,)*) -> Out + 'static
        {
            type Handle = Box<dyn Fn($($t,)*) -> Out + 'static>;

            fn into_raw(self) -> *mut Self::Handle {
                let ptr: Box<Self::Handle> = Box::new(Box::new(self));
                Box::into_raw(ptr)
            }

            #[allow(non_snake_case)]
            fn call<'a>(handle: *mut Self::Handle, args: Vec<ArgValue<'a>>) -> Result<RetValue>
            where
                F: for<'arg> Typed<'arg, ($($t,)*), Out>
            {
                let ($($t,)*) = F::args(args)?;
                let out = unsafe { (*handle)($($t),*) };
                F::ret(out)
            }

            fn drop(ptr: *mut Self::Handle) {
                let bx = unsafe { Box::from_raw(ptr) };
                std::mem::drop(bx)
            }
        }
    }
}

impl_typed_and_to_function!(0;);
impl_typed_and_to_function!(1; A);
impl_typed_and_to_function!(2; A, B);
impl_typed_and_to_function!(3; A, B, C);
impl_typed_and_to_function!(4; A, B, C, D);
impl_typed_and_to_function!(5; A, B, C, D, E);
impl_typed_and_to_function!(6; A, B, C, D, E, G);

#[cfg(test)]
mod tests {
    use super::*;

    fn call<'a, F, I, O>(f: F, args: Vec<ArgValue<'a>>) -> Result<RetValue>
    where
        F: ToFunction<I, O>,
        F: Typed<I, O>,
    {
        F::call(f.into_raw(), args)
    }

    #[test]
    fn test_to_function0() {
        fn zero() -> i32 {
            10
        }
        let _ = zero.to_function();
        let good = call(zero, vec![]).unwrap();
        assert_eq!(i32::try_from(good).unwrap(), 10);
        let bad = call(zero, vec![1.into()]).unwrap_err();
        assert!(matches!(bad, Error::CallFailed(..)));
    }

    #[test]
    fn test_to_function2() {
        fn two_arg(i: i32, j: i32) -> i32 {
            i + j
        }
        let good = call(two_arg, vec![3.into(), 4.into()]).unwrap();
        assert_eq!(i32::try_from(good).unwrap(), 7);
    }
}
