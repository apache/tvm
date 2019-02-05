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
    mem,
    os::raw::{c_char, c_int, c_void},
    ptr, slice, str,
    sync::Mutex,
};

use crate::{ts, ErrorKind, Module, Result, TVMArgValue, TVMRetValue, TVMTypeCode, TVMValue};

lazy_static! {
    static ref GLOBAL_FUNCTIONS: Mutex<BTreeMap<&'static str, Option<Function>>> = {
        let mut out_size = 0 as c_int;
        let name = ptr::null_mut() as *mut c_char;
        let mut out_array = name as *mut _;
        check_call!(ts::TVMFuncListGlobalNames(
            &mut out_size as *mut _,
            &mut out_array
        ));
        let names_list = unsafe { slice::from_raw_parts(out_array, out_size as usize) };
        Mutex::new(
            names_list
                .into_iter()
                .map(|&p| (unsafe { CStr::from_ptr(p).to_str().unwrap() }, None))
                .collect(),
        )
    };
}

/// Wrapper around TVM function handle which includes `is_global`
/// indicating whether the function is global or not, `is_released`
/// to hint dropping the function handle and `is_cloned` showing
/// not to drop a cloned function from Rust side.
/// The value of these fields can be accessed through their respective methods.
#[derive(Debug, Hash)]
pub struct Function {
    pub(crate) handle: ts::TVMFunctionHandle,
    // whether the registered function is global or not.
    is_global: bool,
    // whether the function has been dropped from frontend or not.
    is_released: bool,
    // whether the function has been cloned from frontend or not.
    is_cloned: bool,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    pub(crate) fn new(handle: ts::TVMFunctionHandle, is_global: bool, is_released: bool) -> Self {
        Function {
            handle: handle,
            is_global: is_global,
            is_released: is_released,
            is_cloned: false,
        }
    }

    /// For a given function, it returns a function by name.
    pub fn get<S: AsRef<str>>(name: S, is_global: bool) -> Option<&'static Function> {
        let mut globals = GLOBAL_FUNCTIONS.lock().unwrap();
        globals.get_mut(name.as_ref()).and_then(|maybe_func| {
            if maybe_func.is_none() {
                let name = CString::new(name.as_ref()).unwrap();
                let mut handle = ptr::null_mut() as ts::TVMFunctionHandle;
                check_call!(ts::TVMFuncGetGlobal(
                    name.as_ptr() as *const c_char,
                    &mut handle as *mut _
                ));
                maybe_func.replace(Function::new(
                    handle, is_global, false, /* is_released */
                ));
            }
            unsafe {
                std::mem::transmute::<Option<&Function>, Option<&'static Function>>(
                    maybe_func.as_ref(),
                )
            }
        })
    }

    /// Returns the underlying TVM function handle.
    pub fn handle(&self) -> ts::TVMFunctionHandle {
        self.handle
    }

    /// Returns `true` if the underlying TVM function is global and `false` otherwise.
    pub fn is_global(&self) -> bool {
        self.is_global
    }

    /// Returns `true` if the underlying TVM function has been released
    /// from the frontend and `false` otherwise.
    pub fn is_released(&self) -> bool {
        self.is_released
    }

    /// Returns `true` if the underlying TVM function has been cloned
    /// from the frontend and `false` otherwise.
    pub fn is_cloned(&self) -> bool {
        self.is_cloned
    }
}

impl Clone for Function {
    fn clone(&self) -> Function {
        if !self.is_released && !self.is_cloned {
            Self {
                handle: self.handle,
                is_global: self.is_global,
                is_released: self.is_released,
                is_cloned: true,
            }
        } else {
            Function::new(self.handle, self.is_global, self.is_released)
        }
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        if !self.is_released && !self.is_global && !self.is_cloned {
            check_call!(ts::TVMFuncFree(self.handle));
            self.is_released = true;
        }
    }
}

/// Function builder in order to create and call functions.
///
/// *Note:* Currently TVM functions accept *at most* one return value.
#[derive(Debug, Clone, Default)]
pub struct Builder<'a, 'm> {
    pub func: Option<&'m Function>,
    pub arg_buf: Option<Box<[TVMArgValue<'a>]>>,
    pub ret_buf: Option<TVMRetValue>,
}

impl<'a, 'm> Builder<'a, 'm> {
    pub fn new(
        func: Option<&'m Function>,
        arg_buf: Option<Box<[TVMArgValue<'a>]>>,
        ret_buf: Option<TVMRetValue>,
    ) -> Self {
        Self {
            func,
            arg_buf,
            ret_buf,
        }
    }

    pub fn get_function(&mut self, name: &'m str, is_global: bool) -> &mut Self {
        self.func = Function::get(name, is_global);
        self
    }

    /// Pushes a [`TVMArgValue`] into the function argument buffer.
    pub fn arg<'b, T: ?Sized>(&mut self, arg: &'b T) -> &mut Self
    where
        TVMValue: From<&'b T>,
        TVMTypeCode: From<&'b T>,
    {
        let tvm_arg = TVMArgValue::from(arg);
        if self.arg_buf.is_none() {
            self.arg_buf = Some(Box::new([tvm_arg]));
        } else {
            let new_arg_buf = self.arg_buf.take().map(|bbuf| {
                let mut new_arg_buf = Vec::from(bbuf);
                new_arg_buf.push(tvm_arg);
                let new_len = new_arg_buf.len();
                new_arg_buf.truncate(new_len);
                new_arg_buf.into_boxed_slice()
            });
            self.arg_buf = new_arg_buf;
        }
        self
    }

    /// Pushes multiple [`TVMArgValue`]s into the function argument buffer.
    pub fn args<'b, T: 'b + ?Sized, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = &'b T>,
        TVMValue: From<&'b T>,
        TVMTypeCode: From<&'b T>,
    {
        for arg in args {
            self.arg(&arg);
        }
        self
    }

    /// Sets an output for a function that requirs a mutable output to be provided.
    /// See the `basics` in tests for an example.
    pub fn set_output<'b, T: 'b + ?Sized>(&mut self, arg: &'b mut T) -> Result<&mut Self>
    where
        TVMValue: From<&'b T>,
        TVMTypeCode: From<&'b T>,
    {
        if self.ret_buf.is_none() {
            let tvm_ret =
                unsafe { TVMRetValue::from_tvm_value(TVMValue::from(arg), TVMTypeCode::from(arg)) };
            self.ret_buf = Some(tvm_ret);
        } else {
            bail!(ErrorKind::AtMostOneReturn)
        }
        Ok(self)
    }

    /// Calls the function that created from `Builder`.
    pub fn invoke(&mut self) -> Result<TVMRetValue> {
        self.clone()(())
    }
}

impl<'a, 'm> FnOnce<((),)> for Builder<'a, 'm> {
    type Output = Result<TVMRetValue>;
    extern "rust-call" fn call_once(self, _: ((),)) -> Self::Output {
        if self.func.is_none() {
            bail!("{}", ErrorKind::FunctionNotFound);
        }

        let mut ret_val = unsafe { mem::uninitialized::<ts::TVMValue>() };
        let mut ret_type_code = 0 as c_int;
        if self.arg_buf.is_some() {
            let arg_buf = self.arg_buf?;
            let mut num_args = arg_buf.len();
            let mut values = arg_buf
                .iter()
                .map(|tav| tav.value.inner)
                .collect::<Vec<ts::TVMValue>>();
            let mut tcodes = arg_buf
                .iter()
                .map(|tav| tav.type_code as c_int)
                .collect::<Vec<_>>();

            if self.ret_buf.is_some() {
                num_args = num_args + 1;
                let ret_buf = self.ret_buf?;
                let (ret_val, ret_type_code) = TVMRetValue::into_tvm_value(ret_buf);
                values.append(&mut vec![ret_val.inner]);
                tcodes.append(&mut vec![ret_type_code as c_int]);
            }

            values.truncate(num_args);
            tcodes.truncate(num_args);
            check_call!(ts::TVMFuncCall(
                self.func?.handle,
                values.as_mut_ptr(),
                tcodes.as_mut_ptr(),
                num_args as c_int,
                &mut ret_val as *mut _,
                &mut ret_type_code as *mut _
            ));
        } else {
            check_call!(ts::TVMFuncCall(
                self.func?.handle,
                ptr::null_mut(),
                ptr::null_mut(),
                0 as c_int,
                &mut ret_val as *mut _,
                &mut ret_type_code as *mut _
            ));
        }

        let ret = unsafe {
            TVMRetValue::from_tvm_value(TVMValue::new(ret_val), (ret_type_code as i64).into())
        };
        Ok(ret)
    }
}

/// Converts a [`Function`] to builder. Currently, this is the best way to work with
/// TVM functions.
impl<'a, 'm> From<&'m Function> for Builder<'a, 'm> {
    fn from(func: &'m Function) -> Self {
        Builder::new(Some(func), None, None)
    }
}

/// Converts a mutable reference of a [`Module`] to [`Builder`].
impl<'a, 'm> From<&'m mut Module> for Builder<'a, 'm> {
    fn from(module: &'m mut Module) -> Self {
        Builder::new(module.entry(), None, None)
    }
}

unsafe extern "C" fn tvm_callback(
    args: *mut ts::TVMValue,
    type_codes: *mut c_int,
    num_args: c_int,
    ret: ts::TVMRetValueHandle,
    fhandle: *mut c_void,
) -> c_int {
    // turning off the incorrect linter complaints
    #![allow(unused_assignments)]
    let len = num_args as usize;
    let args_list = slice::from_raw_parts_mut(args, len);
    let type_codes_list = slice::from_raw_parts_mut(type_codes, len);
    let mut local_args: Vec<TVMArgValue> = Vec::new();
    let mut value = mem::uninitialized::<ts::TVMValue>();
    let mut tcode = mem::uninitialized::<c_int>();
    let rust_fn = mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue>>(fhandle);
    for i in 0..len {
        value = args_list[i];
        tcode = type_codes_list[i];
        if tcode == ts::TVMTypeCode_kNodeHandle as c_int
            || tcode == ts::TVMTypeCode_kFuncHandle as c_int
            || tcode == ts::TVMTypeCode_kModuleHandle as c_int
        {
            check_call!(ts::TVMCbArgToReturn(&mut value as *mut _, tcode));
        }
        local_args.push(TVMArgValue::new(
            TVMValue::new(value),
            (tcode as i64).into(),
        ));
    }

    let rv = match rust_fn(local_args.as_slice()) {
        Ok(v) => v,
        Err(msg) => {
            crate::set_last_error(&msg);
            return -1;
        }
    };

    let (ret_val, ret_tcode) = TVMRetValue::into_tvm_value(rv);
    let mut ret_val = ret_val.inner;
    let mut ret_type_code = ret_tcode as c_int;
    check_call!(ts::TVMCFuncSetReturn(
        ret,
        &mut ret_val as *mut _,
        &mut ret_type_code as *mut _,
        1 as c_int
    ));
    0
}

unsafe extern "C" fn tvm_callback_finalizer(fhandle: *mut c_void) {
    let rust_fn = mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue>>(fhandle);
    mem::drop(rust_fn);
}

fn convert_to_tvm_func(f: fn(&[TVMArgValue]) -> Result<TVMRetValue>) -> Function {
    let mut fhandle = ptr::null_mut() as ts::TVMFunctionHandle;
    let resource_handle = f as *mut fn(&[TVMArgValue]) -> Result<TVMRetValue>;
    check_call!(ts::TVMFuncCreateFromCFunc(
        Some(tvm_callback),
        resource_handle as *mut c_void,
        Some(tvm_callback_finalizer),
        &mut fhandle as *mut _
    ));
    Function::new(fhandle, false, false)
}

/// Registers a Rust function with signature
/// `fn(&[TVMArgValue]) -> Result<TVMRetValue>`
/// as a **global TVM packed function** from frontend to TVM backend.
///
/// Use [`register_global_func`] if overriding an existing global TVM function
/// is not required.
///
/// ## Example
///
/// ```
/// use std::convert::TryInto;
///
/// fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
///     let mut ret = 0i64;
///     for arg in args.iter() {
///         let arg: i64 = arg.try_into()?;
///         ret += arg;
///     }
///     let ret_val = TVMRetValue::from(&ret);
///     Ok(ret_val)
/// }
///
/// tvm::function::register(sum, "mysum".to_owned(), false).unwrap();
/// let mut registered = function::Builder::default();
/// registered.get_function("mysum", true);
/// assert!(registered.func.is_some());
/// let ret: i64 = registered.args(&[10, 20, 30]).invoke().unwrap().try_into().unwrap();
/// assert_eq!(ret, 60);
/// ```
pub fn register<S: AsRef<str>>(
    f: fn(&[TVMArgValue]) -> Result<TVMRetValue>,
    name: S,
    override_: bool,
) -> Result<()> {
    let func = convert_to_tvm_func(f);
    let name = CString::new(name.as_ref())?;
    check_call!(ts::TVMFuncRegisterGlobal(
        name.as_ref().as_ptr() as *const c_char,
        func.handle(),
        override_ as c_int
    ));
    mem::forget(name);
    Ok(())
}

/// Convenient macro for registering functions from frontend to backend as global
/// TVM packed functions without overriding. If overriding an existing function is needed
/// use the [`function::register`] function instead.
///
/// ## Example
///
/// ```
/// use std::convert::TryInto;
///
/// register_global_func! {
///     fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
///         let mut ret = 0f64;
///         for arg in args.iter() {
///             let arg: f64 = arg.try_into()?;
///             ret += arg;
///         }
///         let ret_val = TVMRetValue::from(&ret);
///         Ok(ret_val)
///     }
/// }
///
/// let mut registered = function::Builder::default();
/// registered.get_function("sum", true);
/// assert!(registered.func.is_some());
/// let ret: f64 = registered.args(&[10f64, 20f64, 30f64]).invoke().unwrap().try_into().unwrap();
/// assert_eq!(ret, 60f64);
/// ```
#[macro_export]
macro_rules! register_global_func {
    {
        $(#[$m:meta])*
        fn $fn_name:ident($args:ident : &[TVMArgValue]) -> Result<TVMRetValue> {
            $($code:tt)*
        }
    } => {{
        $(#[$m])*
        fn $fn_name($args: &[TVMArgValue]) -> Result<TVMRetValue> {
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
/// ```
/// function::Builder::from(func).arg(&a).arg(&b).invoke();
/// ```
///
/// one can use
///
/// ```
/// call_packed!(func, &a, &b);
/// ```
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

    static CANARY: &str = "module._LoadFromFile";

    #[test]
    fn list_global_func() {
        assert!(GLOBAL_FUNCTIONS.lock().unwrap().contains_key(CANARY));
    }

    #[test]
    fn get_fn() {
        assert!(Function::get(CANARY, true).is_some());
        assert!(Function::get("does not exists!", false).is_none());
    }

    #[test]
    fn provide_args() {
        let mut func = Builder::default();
        func.get_function("tvm.graph_runtime.remote_create", true)
            .args(&[10, 20])
            .arg(&"test".to_owned());
        assert!(func.arg_buf.is_some());
        assert_eq!(func.arg_buf.take().map(|bv| Vec::from(bv).len()), Some(3));
    }
}
