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
use std::sync::Arc;
use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
    ptr, str,
};

use crate::errors::Error;

pub use super::to_function::{RawArgs, ToFunction, Typed};
use crate::object::AsArgValue;
pub use tvm_sys::{ffi, ArgValue, RetValue};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Hash)]
struct FunctionPtr {
    handle: ffi::TVMFunctionHandle,
}

// NB(@jroesch): I think this is ok, need to double check,
// if not we should mutex the pointer or move to Rc.
unsafe impl Send for FunctionPtr {}
unsafe impl Sync for FunctionPtr {}

impl FunctionPtr {
    fn from_raw(handle: ffi::TVMFunctionHandle) -> Self {
        FunctionPtr { handle }
    }
}

impl Drop for FunctionPtr {
    fn drop(&mut self) {
        check_call!(ffi::TVMFuncFree(self.handle));
    }
}

/// An owned thread-safe version of `tvm::PackedFunc` for consumption in Rust.
#[derive(Debug, Hash)]
pub struct Function {
    inner: Arc<FunctionPtr>,
}

impl Function {
    pub(crate) fn from_raw(handle: ffi::TVMFunctionHandle) -> Self {
        Function {
            inner: Arc::new(FunctionPtr::from_raw(handle)),
        }
    }

    pub unsafe fn null() -> Self {
        Function::from_raw(std::ptr::null_mut())
    }

    /// For a given function, it returns a function by name.
    pub fn get<S: AsRef<str>>(name: S) -> Option<Function> {
        let name = CString::new(name.as_ref()).unwrap();
        let mut handle = ptr::null_mut() as ffi::TVMFunctionHandle;

        check_call!(ffi::TVMFuncGetGlobal(
            name.as_ptr() as *const c_char,
            &mut handle as *mut _
        ));

        if handle.is_null() {
            None
        } else {
            Some(Function::from_raw(handle))
        }
    }

    pub fn get_boxed<F, S>(name: S) -> Option<Box<F>>
    where
        S: AsRef<str>,
        F: ?Sized,
        Self: Into<Box<F>>,
    {
        Self::get(name).map(|f| f.into())
    }

    /// Returns the underlying TVM function handle.
    pub fn handle(&self) -> ffi::TVMFunctionHandle {
        self.inner.handle
    }

    /// Calls the function that created from `Builder`.
    pub fn invoke<'a>(&self, arg_buf: Vec<ArgValue<'a>>) -> Result<RetValue> {
        let num_args = arg_buf.len();
        let (mut values, mut type_codes): (Vec<ffi::TVMValue>, Vec<ffi::TVMArgTypeCode>) =
            arg_buf.into_iter().map(|arg| arg.to_tvm_value()).unzip();

        let mut ret_val = ffi::TVMValue { v_int64: 0 };
        let mut ret_type_code = 0i32;

        let ret_code = unsafe {
            ffi::TVMFuncCall(
                self.handle(),
                values.as_mut_ptr() as *mut ffi::TVMValue,
                type_codes.as_mut_ptr() as *mut c_int,
                num_args as c_int,
                &mut ret_val as *mut _,
                &mut ret_type_code as *mut _,
            )
        };

        if ret_code != 0 {
            let raw_error = crate::get_last_error();
            let error = match Error::from_raw_tvm(raw_error) {
                Error::Raw(string) => Error::CallFailed(string),
                e => e,
            };
            return Err(error);
        }

        let rv = RetValue::from_tvm_value(ret_val, ret_type_code as u32);

        Ok(rv)
    }
}

macro_rules! impl_to_fn {
    () => { impl_to_fn!(@impl); };
    ($t:ident, $($ts:ident,)*) => { impl_to_fn!(@impl $t, $($ts,)*); impl_to_fn!($($ts,)*); };
    (@impl $($t:ident,)*) => {
        impl<Err, Out, $($t,)*> From<Function> for Box<dyn Fn($($t,)*) -> Result<Out>>
        where
            Error: From<Err>,
            Out: TryFrom<RetValue, Error = Err>,
            $($t: for<'a> AsArgValue<'a>),*
        {
            fn from(func: Function) -> Self {
                #[allow(non_snake_case)]
                Box::new(move |$($t : $t),*| {
                    let args = vec![ $((&$t).as_arg_value()),* ];
                    Ok(func.invoke(args)?.try_into()?)
                })
            }
        }
    };
}

impl_to_fn!(T1, T2, T3, T4, T5, T6,);

impl Clone for Function {
    fn clone(&self) -> Function {
        Function {
            inner: self.inner.clone(),
        }
    }
}

impl From<Function> for RetValue {
    fn from(func: Function) -> RetValue {
        RetValue::FuncHandle(func.handle())
    }
}

impl TryFrom<RetValue> for Function {
    type Error = Error;

    fn try_from(ret_value: RetValue) -> Result<Function> {
        match ret_value {
            RetValue::FuncHandle(handle) => Ok(Function::from_raw(handle)),
            _ => Err(Error::downcast(
                format!("{:?}", ret_value),
                "FunctionHandle",
            )),
        }
    }
}

impl<'a> From<&'a Function> for ArgValue<'a> {
    fn from(func: &'a Function) -> ArgValue<'a> {
        if func.handle().is_null() {
            ArgValue::Null
        } else {
            ArgValue::FuncHandle(func.handle())
        }
    }
}

impl<'a> TryFrom<ArgValue<'a>> for Function {
    type Error = Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<Function> {
        match arg_value {
            ArgValue::FuncHandle(handle) => Ok(Function::from_raw(handle)),
            _ => Err(Error::downcast(
                format!("{:?}", arg_value),
                "FunctionHandle",
            )),
        }
    }
}

impl<'a> TryFrom<&ArgValue<'a>> for Function {
    type Error = Error;

    fn try_from(arg_value: &ArgValue<'a>) -> Result<Function> {
        match arg_value {
            ArgValue::FuncHandle(handle) => Ok(Function::from_raw(*handle)),
            _ => Err(Error::downcast(
                format!("{:?}", arg_value),
                "FunctionHandle",
            )),
        }
    }
}

/// Registers a Rust function with an arbitrary type signature in
/// the TVM registry.
///
///
/// A function is convertible if and only if its arguments and return types are convertible
/// to and from TVM values respectively.
///
/// Use [`register_override`] if control of overriding existing global TVM function
/// is required, this function will panic if a function is already registered.
///
/// ## Example
///
/// ```
/// # use tvm_rt::{ArgValue, RetValue};
/// # use tvm_rt::function::{Function, Result, register};
///
/// fn sum(x: i64, y: i64, z: i64) -> i64 {
///     x + y + z
/// }
///
/// register(sum, "mysum".to_owned()).unwrap();
/// let func = Function::get("mysum").unwrap();
/// let boxed_fn: Box<dyn Fn(i64, i64, i64) -> Result<i64>> = func.into();
/// let ret = boxed_fn(10, 20, 30).unwrap();
/// assert_eq!(ret, 60);
/// ```
pub fn register<F, I, O, S: Into<String>>(f: F, name: S) -> Result<()>
where
    F: ToFunction<I, O>,
    F: Typed<I, O>,
{
    register_override(f, name, false)
}

/// Register a function with explicit control over whether to override an existing registration or not.
///
/// See `register` for more details on how to use the registration API.
pub fn register_override<F, I, O, S: Into<String>>(f: F, name: S, override_: bool) -> Result<()>
where
    F: ToFunction<I, O>,
    F: Typed<I, O>,
{
    let func = f.to_function();
    let name = name.into();
    // Not sure about this code
    let handle = func.handle();
    let name = CString::new(name)?;
    check_call!(ffi::TVMFuncRegisterGlobal(
        name.into_raw(),
        handle,
        override_ as c_int
    ));

    Ok(())
}

pub fn register_untyped<S: Into<String>>(
    f: for<'a> fn(Vec<ArgValue<'a>>) -> Result<RetValue>,
    name: S,
    override_: bool,
) -> Result<()> {
    //TODO(@jroesch): can we unify the untpyed and typed registration functions.
    let func = ToFunction::<RawArgs, RetValue>::to_function(f);
    let name = name.into();
    // Not sure about this code
    let handle = func.handle();
    let name = CString::new(name)?;
    check_call!(ffi::TVMFuncRegisterGlobal(
        name.into_raw(),
        handle,
        override_ as c_int
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Function;

    static CANARY: &str = "runtime.ModuleLoadFromFile";

    #[test]
    fn get_fn() {
        assert!(Function::get(CANARY).is_some());
        assert!(Function::get("does not exists!").is_none());
    }

    #[test]
    fn register_and_call_closure0() {
        use crate::function;
        use function::Result;

        fn constfn() -> i64 {
            return 10;
        }

        function::register_override(constfn, "constfn".to_owned(), true).unwrap();

        let func = Function::get_boxed::<dyn Fn() -> Result<i32>, _>("constfn").unwrap();
        let ret = func().unwrap();
        assert_eq!(ret, 10);
    }

    #[test]
    fn register_and_call_closure1() {
        use crate::function::{self};

        fn ident(x: i64) -> i64 {
            return x;
        }

        function::register_override(ident, "ident".to_owned(), true).unwrap();
        let func = Function::get_boxed::<dyn Fn(i32) -> Result<i32>, _>("ident").unwrap();
        assert_eq!(func(60).unwrap(), 60);
    }
}
