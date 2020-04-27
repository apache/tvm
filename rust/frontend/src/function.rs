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
    collections::BTreeMap,
    ffi::{CStr, CString},
    mem::{self, MaybeUninit},
    os::raw::{c_char, c_int, c_void},
    ptr, slice, str,
    sync::Mutex,
};

use failure::Error;

use crate::{errors, ffi, Module, TVMArgValue, TVMRetValue};

lazy_static! {
    static ref GLOBAL_FUNCTIONS: Mutex<BTreeMap<&'static str, Option<Function>>> = {
        let mut out_size = 0 as c_int;
        let mut names_ptr = ptr::null_mut() as *mut *const c_char;
        check_call!(ffi::TVMFuncListGlobalNames(
            &mut out_size as *mut _,
            &mut names_ptr as *mut _,
        ));
        let names_list = unsafe { slice::from_raw_parts(names_ptr, out_size as usize) };
        let names_list = names_list
            .iter()
            .map(|&p| (unsafe { CStr::from_ptr(p).to_str().unwrap() }, None))
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

/// Function builder in order to create and call functions.
///
/// *Note:* Currently TVM functions accept *at most* one return value.
#[derive(Default)]
pub struct Builder<'a, 'm> {
    pub func: Option<&'m Function>,
    pub arg_buf: Vec<TVMArgValue<'a>>,
    pub ret_buf: Option<TVMRetValue>,
}

impl<'a, 'm> Builder<'a, 'm> {
    pub fn new(
        func: Option<&'m Function>,
        arg_buf: Vec<TVMArgValue<'a>>,
        ret_buf: Option<TVMRetValue>,
    ) -> Self {
        Self {
            func,
            arg_buf,
            ret_buf,
        }
    }

    pub fn get_function(&mut self, name: &'m str) -> &mut Self {
        self.func = Function::get(name);
        self
    }

    /// Pushes a [`TVMArgValue`] into the function argument buffer.
    pub fn arg<T: 'a>(&mut self, arg: T) -> &mut Self
    where
        TVMArgValue<'a>: From<T>,
    {
        self.arg_buf.push(arg.into());
        self
    }

    /// Pushes multiple [`TVMArgValue`]s into the function argument buffer.
    pub fn args<T: 'a, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a T>,
        TVMArgValue<'a>: From<&'a T>,
    {
        args.into_iter().for_each(|arg| {
            self.arg(&arg);
        });
        self
    }

    /// Sets an output for a function that requirs a mutable output to be provided.
    /// See the `basics` in tests for an example.
    pub fn set_output<T>(&mut self, ret: T) -> &mut Self
    where
        TVMRetValue: From<T>,
    {
        self.ret_buf = Some(ret.into());
        self
    }

    /// Calls the function that created from `Builder`.
    pub fn invoke(&mut self) -> Result<TVMRetValue, Error> {
        #![allow(unused_unsafe)]
        ensure!(self.func.is_some(), errors::FunctionNotFoundError);

        let num_args = self.arg_buf.len();
        let (mut values, mut type_codes): (Vec<ffi::TVMValue>, Vec<ffi::TVMTypeCode>) =
            self.arg_buf.iter().map(|arg| arg.to_tvm_value()).unzip();

        let mut ret_val = unsafe { MaybeUninit::uninit().assume_init() };
        let mut ret_type_code = 0i32;
        check_call!(ffi::TVMFuncCall(
            self.func.ok_or(errors::FunctionNotFoundError)?.handle,
            values.as_mut_ptr(),
            type_codes.as_mut_ptr() as *mut i32,
            num_args as c_int,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _
        ));

        Ok(unsafe { TVMRetValue::from_tvm_value(ret_val, ret_type_code as u32) })
    }
}

/// Converts a [`Function`] to builder. Currently, this is the best way to work with
/// TVM functions.
impl<'a, 'm> From<&'m Function> for Builder<'a, 'm> {
    fn from(func: &'m Function) -> Self {
        Builder::new(Some(func), Vec::new(), None)
    }
}

/// Converts a mutable reference of a [`Module`] to [`Builder`].
impl<'a, 'm> From<&'m mut Module> for Builder<'a, 'm> {
    fn from(module: &'m mut Module) -> Self {
        Builder::new(module.entry(), Vec::new(), None)
    }
}

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
    let mut local_args: Vec<TVMArgValue> = Vec::new();
    let mut value = MaybeUninit::uninit().assume_init();
    let mut tcode = MaybeUninit::uninit().assume_init();
    let rust_fn =
        mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>>(fhandle);
    for i in 0..len {
        value = args_list[i];
        tcode = type_codes_list[i];
        if tcode == ffi::TVMTypeCode_kTVMObjectHandle as c_int
            || tcode == ffi::TVMTypeCode_kTVMPackedFuncHandle as c_int
            || tcode == ffi::TVMTypeCode_kTVMModuleHandle as c_int
        {
            check_call!(ffi::TVMCbArgToReturn(
                &mut value as *mut _,
                &mut tcode as *mut _
            ));
        }
        local_args.push(TVMArgValue::from_tvm_value(value, tcode as u32));
    }

    let rv = match rust_fn(local_args.as_slice()) {
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

unsafe extern "C" fn tvm_callback_finalizer(fhandle: *mut c_void) {
    let _rust_fn =
        mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>>(fhandle);
    // XXX: give converted functions lifetimes so they're not called after use
}

fn convert_to_tvm_func(f: fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>) -> Function {
    let mut fhandle = ptr::null_mut() as ffi::TVMFunctionHandle;
    let resource_handle = f as *mut fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>;
    check_call!(ffi::TVMFuncCreateFromCFunc(
        Some(tvm_callback),
        resource_handle as *mut c_void,
        Some(tvm_callback_finalizer),
        &mut fhandle as *mut _
    ));
    Function::new(fhandle)
}

/// Registers a Rust function with signature
/// `fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>`
/// as a **global TVM packed function** from frontend to TVM backend.
///
/// Use [`register_global_func`] if overriding an existing global TVM function
/// is not required.
///
/// ## Example
///
/// ```
/// # use tvm_frontend::{TVMArgValue, function, TVMRetValue};
/// # use tvm_frontend::function::Builder;
/// # use failure::Error;
/// use std::convert::TryInto;
///
/// fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
///     let mut ret = 0i64;
///     for arg in args.iter() {
///         let arg: i64 = arg.try_into()?;
///         ret += arg;
///     }
///     let ret_val = TVMRetValue::from(ret);
///     Ok(ret_val)
/// }
///
/// function::register(sum, "mysum".to_owned(), false).unwrap();
/// let mut registered = Builder::default();
/// registered.get_function("mysum");
/// assert!(registered.func.is_some());
/// let ret: i64 = registered.args(&[10, 20, 30]).invoke().unwrap().try_into().unwrap();
/// assert_eq!(ret, 60);
/// ```
pub fn register<S: AsRef<str>>(
    f: fn(&[TVMArgValue]) -> Result<TVMRetValue, Error>,
    name: S,
    override_: bool,
) -> Result<(), Error> {
    let func = convert_to_tvm_func(f);
    let name = CString::new(name.as_ref())?;
    check_call!(ffi::TVMFuncRegisterGlobal(
        name.into_raw(),
        func.handle(),
        override_ as c_int
    ));
    Ok(())
}

/// Convenient macro for registering functions from frontend to backend as global
/// TVM packed functions without overriding. If overriding an existing function is needed
/// use the [`function::register`] function instead.
///
/// ## Example
///
/// ```
/// # use std::convert::TryInto;
/// # use tvm_frontend::{register_global_func, TVMArgValue, TVMRetValue};
/// # use failure::Error;
/// # use tvm_frontend::function::Builder;
///
/// register_global_func! {
///     fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
///         let mut ret = 0f64;
///         for arg in args.iter() {
///             let arg: f64 = arg.try_into()?;
///             ret += arg;
///         }
///         let ret_val = TVMRetValue::from(ret);
///         Ok(ret_val)
///     }
/// }
///
/// let mut registered = Builder::default();
/// registered.get_function("sum");
/// assert!(registered.func.is_some());
/// let ret: f64 = registered.args(&[10f64, 20f64, 30f64]).invoke().unwrap().try_into().unwrap();
/// assert_eq!(ret, 60f64);
/// ```
#[macro_export]
macro_rules! register_global_func {
    {
        $(#[$m:meta])*
        fn $fn_name:ident($args:ident : &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            $($code:tt)*
        }
    } => {{
        $(#[$m])*
        fn $fn_name($args: &[TVMArgValue]) -> Result<TVMRetValue, Error> {
            $($code)*
        }

        $crate::function::register($fn_name, stringify!($fn_name).to_owned(), false).unwrap();
    }}
}

/// Convenient macro for calling TVM packed functions by providing a
/// function identifier and some arguments. This macro outputs a `Result` type
/// and let user to perform proper error handling.
///
/// **Note**: this macro does *not* expect an outside mutable output. To
/// set mutable output use [`set_output`] directly in the builder pattern.
///
/// [`set_output`]:function/struct.Builder.html#method.set_output
///
/// ## Example
///
/// Instead of
///
/// # TODO(@jroesch): replace with working example
/// # use tvm_frontend::function::Builder;
/// Builder::from(func).arg(&a).arg(&b).invoke();
///
/// one can use
///
/// # use tvm_frontend::call_packed;
/// call_packed!(func, &a, &b);
#[macro_export]
macro_rules! call_packed {
    ($fn_name:expr, $($arg:expr),*) => {{
        let mut builder = $crate::function::Builder::from($fn_name);
        $(
            builder.arg($arg);
        )*
        builder.invoke()
    }}
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn provide_args() {
        let str_arg = CString::new("test").unwrap();
        let mut func = Builder::default();
        func.get_function("tvm.graph_runtime.remote_create")
            .arg(10)
            .arg(20)
            .arg(str_arg.as_c_str());
        assert_eq!(func.arg_buf.len(), 3);
    }
}
