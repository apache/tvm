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

use lazy_static::lazy_static;
use std::convert::TryFrom;
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString},
    mem::{self, MaybeUninit},
    os::raw::{c_char, c_int},
    ptr, slice, str,
    sync::Mutex,
};

pub use tvm_sys::{ffi, ArgValue, RetValue};

use crate::errors::Error;

use super::to_boxed_fn::ToBoxedFn;
use super::to_function::{ToFunction, Typed};

pub type Result<T> = std::result::Result<T, Error>;

lazy_static! {
    static ref GLOBAL_FUNCTIONS: Mutex<BTreeMap<String, Option<Function>>> = {
        let mut out_size = 0 as c_int;
        let mut names_ptr = ptr::null_mut() as *mut *const c_char;
        check_call!(ffi::TVMFuncListGlobalNames(
            &mut out_size as *mut _,
            &mut names_ptr as *mut _,
        ));
        let names_list = unsafe { slice::from_raw_parts(names_ptr, out_size as usize) };

        let names_list: Vec<String> =
            names_list
            .iter()
            .map(|&p| unsafe { CStr::from_ptr(p).to_str().unwrap().into() })
            .collect();

        // println!("{:?}", &names_list);

        let names_list = names_list
            .into_iter()
            .map(|p| (p, None))
            .collect();

        Mutex::new(names_list)
    };
}

/// Wrapper around TVM function handle which includes `is_global`
/// indicating whether the function is global or not, and `is_cloned` showing
/// not to drop a cloned function from Rust side.
/// The value of these fields can be accessed through their respective methods.
#[derive(Debug, Hash)]
pub struct Function {
    pub(crate) handle: ffi::TVMFunctionHandle,
    // whether the registered function is global or not.
    is_global: bool,
    // whether the function has been cloned from frontend or not.
    is_cloned: bool,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub(crate) fn new(handle: ffi::TVMFunctionHandle) -> Self {
        Function {
            handle,
            is_global: false,
            is_cloned: false,
        }
    }

    /// For a given function, it returns a function by name.
    pub fn get<S: AsRef<str>>(name: S) -> Option<&'static Function> {
        let mut globals = GLOBAL_FUNCTIONS.lock().unwrap();
        globals.get_mut(name.as_ref()).and_then(|maybe_func| {
            if maybe_func.is_none() {
                let name = CString::new(name.as_ref()).unwrap();
                let mut handle = ptr::null_mut() as ffi::TVMFunctionHandle;
                check_call!(ffi::TVMFuncGetGlobal(
                    name.as_ptr() as *const c_char,
                    &mut handle as *mut _
                ));
                maybe_func.replace(Function {
                    handle,
                    is_global: true,
                    is_cloned: false,
                });
            }

            unsafe {
                mem::transmute::<Option<&Function>, Option<&'static Function>>(maybe_func.as_ref())
            }
        })
    }

    /// Returns the underlying TVM function handle.
    pub fn handle(&self) -> ffi::TVMFunctionHandle {
        self.handle
    }

    /// Returns `true` if the underlying TVM function is global and `false` otherwise.
    pub fn is_global(&self) -> bool {
        self.is_global
    }

    /// Returns `true` if the underlying TVM function has been cloned
    /// from the frontend and `false` otherwise.
    pub fn is_cloned(&self) -> bool {
        self.is_cloned
    }

    /// Calls the function that created from `Builder`.
    pub fn invoke<'a>(&self, arg_buf: Vec<ArgValue<'a>>) -> Result<RetValue> {
        let num_args = arg_buf.len();
        let (mut values, mut type_codes): (Vec<ffi::TVMValue>, Vec<ffi::TVMTypeCode>) =
            arg_buf.iter().map(|arg| arg.to_tvm_value()).unzip();

        let mut ret_val = unsafe { MaybeUninit::uninit().assume_init() };
        let mut ret_type_code = 0i32;
        check_call!(ffi::TVMFuncCall(
            self.handle,
            values.as_mut_ptr(),
            type_codes.as_mut_ptr() as *mut i32,
            num_args as c_int,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _
        ));

        Ok(RetValue::from_tvm_value(ret_val, ret_type_code as u32))
    }

    pub fn to_boxed_fn<F: ?Sized>(&'static self) -> Box<F>
    where
        F: ToBoxedFn,
    {
        F::to_boxed_fn(self)
    }
}

impl Clone for Function {
    fn clone(&self) -> Function {
        Self {
            handle: self.handle,
            is_global: self.is_global,
            is_cloned: true,
        }
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        if !self.is_global && !self.is_cloned {
            check_call!(ffi::TVMFuncFree(self.handle));
        }
    }
}

impl From<Function> for RetValue {
    fn from(func: Function) -> RetValue {
        RetValue::FuncHandle(func.handle)
    }
}

impl TryFrom<RetValue> for Function {
    type Error = Error;

    fn try_from(ret_value: RetValue) -> Result<Function> {
        match ret_value {
            RetValue::FuncHandle(handle) => Ok(Function::new(handle)),
            _ => Err(Error::downcast(
                format!("{:?}", ret_value),
                "FunctionHandle",
            )),
        }
    }
}

impl<'a> From<Function> for ArgValue<'a> {
    fn from(func: Function) -> ArgValue<'a> {
        ArgValue::FuncHandle(func.handle)
    }
}

impl<'a> TryFrom<ArgValue<'a>> for Function {
    type Error = Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<Function> {
        match arg_value {
            ArgValue::FuncHandle(handle) => Ok(Function::new(handle)),
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
            ArgValue::FuncHandle(handle) => Ok(Function::new(*handle)),
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
/// let boxed_fn = func.to_boxed_fn::<dyn Fn(i64, i64, i64) -> Result<i64>>();
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
    let mut globals = GLOBAL_FUNCTIONS.lock().unwrap();
    // Not sure about this code
    let handle = func.handle();
    globals.insert(name.clone(), Some(func));
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

    // #[test]
    // fn list_global_func() {
    //     assert!(GLOBAL_FUNCTIONS.lock().unwrap().contains_key(CANARY));
    // }

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
        let func = Function::get("constfn").unwrap();
        let func = func.to_boxed_fn::<dyn Fn() -> Result<i32>>();
        let ret = func().unwrap();
        assert_eq!(ret, 10);
    }

    // #[test]
    // fn register_and_call_closure1() {
    //     use crate::function::{self};

    //     fn ident(x: i64) -> i64 {
    //         return x;
    //     }

    //     function::register_override(ident, "ident".to_owned(), false).unwrap();
    //     let func = Function::get("ident").unwrap();
    //     let func = func.to_boxed_fn::<dyn Fn(i32) -> Result<i32>>();
    //     assert_eq!(func(60).unwrap(), 60);
    // }
}
