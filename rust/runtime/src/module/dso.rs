use std::{
    cell::RefCell,
    collections::HashMap,
    os::raw::{c_int, c_void},
};

use tvm_common::{ffi::BackendPackedCFunc, packed_func::PackedFunc};

use crate::{
    threading::{TVMBackendParallelBarrier, TVMBackendParallelLaunch},
    workspace::{TVMBackendAllocWorkspace, TVMBackendFreeWorkspace},
    TVMAPISetLastError,
};

use super::Module;

/// A module backed by a Dynamic Shared Object (dylib).
pub struct DsoModule<'a> {
    lib: libloading::Library,
    packed_funcs: RefCell<HashMap<String, &'a (dyn PackedFunc)>>,
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
    pub fn new<P: AsRef<std::ffi::OsStr>>(filename: P) -> Result<Self, failure::Error> {
        let lib = libloading::Library::new(filename)?;

        init_context_func!(
            lib,
            (TVMAPISetLastError, extern "C" fn(*const i8)),
            (
                TVMBackendAllocWorkspace,
                extern "C" fn(c_int, c_int, u64, c_int, c_int) -> *mut c_void
            ),
            (
                TVMBackendFreeWorkspace,
                extern "C" fn(c_int, c_int, *mut c_void) -> c_int
            ),
            (
                TVMBackendParallelLaunch,
                extern "C" fn(crate::threading::FTVMParallelLambda, *const c_void, usize) -> c_int
            ),
            (
                TVMBackendParallelBarrier,
                extern "C" fn(usize, *const tvm_common::ffi::TVMParallelGroupEnv)
            ),
        );

        Ok(Self {
            lib,
            packed_funcs: RefCell::new(HashMap::new()),
        })
    }
}

const TVM_MAIN: &'static [u8] = b"__tvm_main__";

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

        self.packed_funcs.borrow().get(name).map(|f| *f)
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
