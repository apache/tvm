use std::{any::Any, convert::TryFrom, marker::PhantomData, os::raw::c_void};

use failure::Error;

pub use crate::ffi::TVMValue;
use crate::ffi::*;

pub trait PackedFunc =
    Fn(&[TVMArgValue]) -> Result<TVMRetValue, crate::errors::FuncCallError> + Send + Sync;

/// Calls a packed function and returns a `TVMRetValue`.
///
/// # Example
///
/// `call_packed!(my_tvm_func, &mut arg1, &mut arg2)`
#[macro_export]
macro_rules! call_packed {
  ($fn:expr, $($args:expr),+) => {
    $fn(&[$($args.into(),)+])
  };
  ($fn:expr) => {
    $fn(&Vec::new())
  };
}

/// A borrowed TVMPODValue. Can be constructed using `into()` but the preferred way
/// to obtain a `TVMArgValue` is automatically via `call_packed!`.
#[derive(Clone, Copy)]
pub struct TVMArgValue<'a> {
    pub _lifetime: PhantomData<&'a ()>,
    pub value: TVMValue,
    pub type_code: i64,
}

impl<'a> TVMArgValue<'a> {
    pub fn new(value: TVMValue, type_code: i64) -> Self {
        TVMArgValue {
            _lifetime: PhantomData,
            value: value,
            type_code: type_code,
        }
    }
}

#[macro_export]
macro_rules! ensure_type {
    ($val:ident, $expected_type_code:expr) => {
        ensure!(
            $val.type_code == $expected_type_code as i64,
            $crate::errors::ValueDowncastError::new(
                $val.type_code as i64,
                $expected_type_code as i64
            )
        );
    };
}

/// Creates a conversion to a `TVMArgValue` for a primitive type and DLDataTypeCode.
macro_rules! impl_prim_tvm_arg {
    ($type_code:ident, $field:ident, $field_type:ty, [ $( $type:ty ),+ ] ) => {
        $(
            impl From<$type> for TVMArgValue<'static> {
                fn from(val: $type) -> Self {
                    TVMArgValue {
                        value: TVMValue { $field: val as $field_type },
                        type_code: $type_code as i64,
                        _lifetime: PhantomData,
                    }
                }
            }
            impl<'a> From<&'a $type> for TVMArgValue<'a> {
                fn from(val: &'a $type) -> Self {
                    TVMArgValue {
                        value: TVMValue {
                            $field: val.to_owned() as $field_type,
                        },
                        type_code: $type_code as i64,
                        _lifetime: PhantomData,
                    }
                }
            }
            impl<'a> TryFrom<TVMArgValue<'a>> for $type {
              type Error = Error;
                fn try_from(val: TVMArgValue<'a>) -> Result<Self, Self::Error> {
                    ensure_type!(val, $type_code);
                    Ok(unsafe { val.value.$field as $type })
                }
            }

            impl<'a> TryFrom<&TVMArgValue<'a>> for $type {
              type Error = Error;
                fn try_from(val: &TVMArgValue<'a>) -> Result<Self, Self::Error> {
                    ensure_type!(val, $type_code);
                    Ok(unsafe { val.value.$field as $type })
                }
            }
        )+
    };
}

impl_prim_tvm_arg!(DLDataTypeCode_kDLFloat, v_float64, f64, [f32, f64]);
impl_prim_tvm_arg!(
    DLDataTypeCode_kDLInt,
    v_int64,
    i64,
    [i8, i16, i32, i64, isize]
);
impl_prim_tvm_arg!(
    DLDataTypeCode_kDLUInt,
    v_int64,
    i64,
    [u8, u16, u32, u64, usize]
);

#[cfg(feature = "bindings")]
// only allow this in bindings because pure-rust can't take ownership of leaked CString
impl<'a> From<&String> for TVMArgValue<'a> {
    fn from(string: &String) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_str: std::ffi::CString::new(string.clone()).unwrap().into_raw(),
            },
            type_code: TVMTypeCode_kStr as i64,
            _lifetime: PhantomData,
        }
    }
}

impl<'a> From<&std::ffi::CString> for TVMArgValue<'a> {
    fn from(string: &std::ffi::CString) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_str: string.as_ptr(),
            },
            type_code: TVMTypeCode_kStr as i64,
            _lifetime: PhantomData,
        }
    }
}

impl<'a> TryFrom<TVMArgValue<'a>> for &str {
    type Error = Error;
    fn try_from(arg: TVMArgValue<'a>) -> Result<Self, Self::Error> {
        ensure_type!(arg, TVMTypeCode_kStr);
        Ok(unsafe { std::ffi::CStr::from_ptr(arg.value.v_handle as *const i8) }.to_str()?)
    }
}

impl<'a> TryFrom<&TVMArgValue<'a>> for &str {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self, Self::Error> {
        ensure_type!(arg, TVMTypeCode_kStr);
        Ok(unsafe { std::ffi::CStr::from_ptr(arg.value.v_handle as *const i8) }.to_str()?)
    }
}

/// Creates a conversion to a `TVMArgValue` for an object handle.
impl<'a, T> From<*const T> for TVMArgValue<'a> {
    fn from(ptr: *const T) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_handle: ptr as *mut T as *mut c_void,
            },
            type_code: TVMTypeCode_kArrayHandle as i64,
            _lifetime: PhantomData,
        }
    }
}

/// Creates a conversion to a `TVMArgValue` for a mutable object handle.
impl<'a, T> From<*mut T> for TVMArgValue<'a> {
    fn from(ptr: *mut T) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_handle: ptr as *mut c_void,
            },
            type_code: TVMTypeCode_kHandle as i64,
            _lifetime: PhantomData,
        }
    }
}

impl<'a> From<&'a mut DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a mut DLTensor) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_handle: arr as *mut _ as *mut c_void,
            },
            type_code: TVMTypeCode_kArrayHandle as i64,
            _lifetime: PhantomData,
        }
    }
}

impl<'a> From<&'a DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a DLTensor) -> Self {
        TVMArgValue {
            value: TVMValue {
                v_handle: arr as *const _ as *mut DLTensor as *mut c_void,
            },
            type_code: TVMTypeCode_kArrayHandle as i64,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for TVMType {
    type Error = Error;
    fn try_from(arg: &'a TVMArgValue<'v>) -> Result<Self, Self::Error> {
        ensure_type!(arg, TVMTypeCode_kTVMType);
        Ok(unsafe { arg.value.v_type.into() })
    }
}

/// An owned TVMPODValue. Can be converted from a variety of primitive and object types.
/// Can be downcasted using `try_from` if it contains the desired type.
///
/// # Example
///
/// ```
/// let a = 42u32;
/// let b: i64 = TVMRetValue::from(a).try_into().unwrap();
///
/// let s = "hello, world!";
/// let t: TVMRetValue = s.into();
/// assert_eq!(String::try_from(t).unwrap(), s);
/// ```
pub struct TVMRetValue {
    pub value: TVMValue,
    pub box_value: Box<Any>,
    pub type_code: i64,
}

impl TVMRetValue {
    pub fn from_tvm_value(value: TVMValue, type_code: i64) -> Self {
        Self {
            value,
            type_code,
            box_value: box (),
        }
    }

    pub fn into_tvm_value(self) -> (TVMValue, TVMTypeCode) {
        (self.value, self.type_code as TVMTypeCode)
    }
}

impl Default for TVMRetValue {
    fn default() -> Self {
        TVMRetValue {
            value: TVMValue { v_int64: 0 as i64 },
            type_code: 0,
            box_value: box (),
        }
    }
}

macro_rules! impl_pod_ret_value {
    ($code:expr, [ $( $ty:ty ),+ ] ) => {
        $(
            impl From<$ty> for TVMRetValue {
                fn from(val: $ty) -> Self {
                    Self {
                        value: val.into(),
                        type_code: $code as i64,
                        box_value: box (),
                    }
                }
            }

            impl TryFrom<TVMRetValue> for $ty {
              type Error = Error;
                fn try_from(ret: TVMRetValue) -> Result<$ty, Self::Error> {
                    ensure_type!(ret, $code);
                    Ok(ret.value.into())
                }
            }
        )+
    };
}

impl_pod_ret_value!(DLDataTypeCode_kDLInt, [i8, i16, i32, i64, isize]);
impl_pod_ret_value!(DLDataTypeCode_kDLUInt, [u8, u16, u32, u64, usize]);
impl_pod_ret_value!(DLDataTypeCode_kDLFloat, [f32, f64]);
impl_pod_ret_value!(TVMTypeCode_kTVMType, [TVMType]);
impl_pod_ret_value!(TVMTypeCode_kTVMContext, [TVMContext]);

impl TryFrom<TVMRetValue> for String {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<String, Self::Error> {
        ensure_type!(ret, TVMTypeCode_kStr);
        let cs = unsafe { std::ffi::CString::from_raw(ret.value.v_handle as *mut i8) };
        let ret_str = cs.clone().into_string();
        if cfg!(feature = "bindings") {
            std::mem::forget(cs); // TVM C++ takes ownership of CString. (@see TVMFuncCall)
        }
        Ok(ret_str?)
    }
}

impl From<String> for TVMRetValue {
    fn from(s: String) -> Self {
        let cs = std::ffi::CString::new(s).unwrap();
        Self {
            value: TVMValue {
                v_str: cs.into_raw() as *mut i8,
            },
            box_value: box (),
            type_code: TVMTypeCode_kStr as i64,
        }
    }
}
