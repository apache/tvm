//! This module provides the the wrapped `TVMValue`, `TVMArgValue` and `TVMRetValue`
//! required for using TVM functions.

use std::{
    any::Any,
    convert::TryFrom,
    ffi::{CStr, CString},
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem,
    ops::Deref,
    os::raw::{c_char, c_void},
};

#[cfg(feature = "runtime")]
use ffi::runtime::TVMValue as _TVMValue;

#[cfg(feature = "frontend")]
use ffi::ts::TVMValue as _TVMValue;

use errors::*;

use ty::TVMTypeCode;

/// Wrapped TVMValue type.
#[derive(Clone, Copy)]
pub struct TVMValue {
    pub inner: _TVMValue,
}

impl TVMValue {
    /// Creates TVMValue from the raw part.
    pub fn new(inner: _TVMValue) -> Self {
        TVMValue { inner }
    }

    pub(crate) fn into_raw(self) -> _TVMValue {
        self.inner
    }
}

impl Debug for TVMValue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe {
            write!(
                f,
                "TVMValue: [v_int64: {:?}], [v_float64: {:?}], [v_handle: {:?}],\
                 [v_str: {:?}]",
                self.inner.v_int64, self.inner.v_float64, self.inner.v_handle, self.inner.v_str
            )
        }
    }
}

impl Deref for TVMValue {
    type Target = _TVMValue;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

macro_rules! impl_prim_val {
    ($type:ty, $field:ident, $cast:ty) => {
        impl From<$type> for TVMValue {
            fn from(arg: $type) -> Self {
                let inner = _TVMValue {
                    $field: arg as $cast,
                };
                Self::new(inner)
            }
        }

        impl<'a> From<&'a $type> for TVMValue {
            fn from(arg: &$type) -> Self {
                let inner = _TVMValue {
                    $field: *arg as $cast,
                };
                Self::new(inner)
            }
        }

        impl<'a> From<&'a mut $type> for TVMValue {
            fn from(arg: &mut $type) -> Self {
                let inner = _TVMValue {
                    $field: *arg as $cast,
                };
                Self::new(inner)
            }
        }

        impl TryFrom<TVMValue> for $type {
            type Error = Error;
            fn try_from(val: TVMValue) -> Result<Self> {
                Ok(unsafe { val.inner.$field as $type })
            }
        }

        impl<'a> TryFrom<&'a TVMValue> for $type {
            type Error = Error;
            fn try_from(val: &TVMValue) -> Result<Self> {
                Ok(unsafe { val.into_raw().$field as $type })
            }
        }

        impl<'a> TryFrom<&'a mut TVMValue> for $type {
            type Error = Error;
            fn try_from(val: &mut TVMValue) -> Result<Self> {
                Ok(unsafe { val.into_raw().$field as $type })
            }
        }
    };
}

impl_prim_val!(isize, v_int64, i64);
impl_prim_val!(i64, v_int64, i64);
impl_prim_val!(i32, v_int64, i64);
impl_prim_val!(i16, v_int64, i64);
impl_prim_val!(i8, v_int64, i64);
impl_prim_val!(usize, v_int64, i64);
impl_prim_val!(u64, v_int64, i64);
impl_prim_val!(u32, v_int64, i64);
impl_prim_val!(u16, v_int64, i64);
impl_prim_val!(u8, v_int64, i64);

impl_prim_val!(f64, v_float64, f64);
impl_prim_val!(f32, v_float64, f64);

impl<'a> From<&'a str> for TVMValue {
    fn from(arg: &str) -> TVMValue {
        let arg = CString::new(arg).unwrap();
        let inner = _TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(inner)
    }
}

impl<'a> From<&'a String> for TVMValue {
    fn from(arg: &String) -> TVMValue {
        let arg = CString::new(arg.as_bytes()).unwrap();
        let inner = _TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(inner)
    }
}

impl<'a> From<&'a CString> for TVMValue {
    fn from(arg: &CString) -> TVMValue {
        let arg = arg.to_owned();
        let inner = _TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(inner)
    }
}

impl<'a> From<&'a [u8]> for TVMValue {
    fn from(arg: &[u8]) -> TVMValue {
        let arg = arg.to_owned();
        let inner = _TVMValue {
            v_handle: &arg as *const _ as *mut c_void,
        };
        mem::forget(arg);
        Self::new(inner)
    }
}

/// Captures both `TVMValue` and `TVMTypeCode` needed for TVM function.
/// The preferred way to obtain a `TVMArgValue` is automatically via `call_packed!`.
/// or in the frontend crate, with `function::Builder`. Checkout the methods for conversions.
///
/// ## Example
///
/// ```
/// let s = "hello".to_string();
/// let arg = TVMArgValue::from(&s);
/// let tvm: String = arg.try_into().unwrap();
/// assert_eq!(arg, s);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TVMArgValue<'a> {
    /// The wrapped TVMValue
    pub value: TVMValue,
    /// The matching type code.
    pub type_code: TVMTypeCode,
    /// This is only exposed to runtime and frontend crates and is not meant to be used directly.
    pub lifetime: PhantomData<&'a ()>,
}

impl<'a> TVMArgValue<'a> {
    pub fn new(value: TVMValue, type_code: TVMTypeCode) -> Self {
        TVMArgValue {
            value: value,
            type_code: type_code,
            lifetime: PhantomData,
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for i64 {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if (arg.type_code == TVMTypeCode::kDLInt)
            | (arg.type_code == TVMTypeCode::kDLUInt)
            | (arg.type_code == TVMTypeCode::kNull)
        {
            Ok(unsafe { arg.value.inner.v_int64 })
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(i64).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for f64 {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kDLFloat {
            Ok(unsafe { arg.value.inner.v_float64 })
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(f64).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for String {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kStr {
            let ret_str = unsafe {
                match CStr::from_ptr(arg.value.inner.v_str).to_str() {
                    Ok(s) => s,
                    Err(_) => "Invalid UTF-8 message",
                }
            };
            Ok(ret_str.to_string())
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(String).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

/// Main way to create a TVMArgValue from suported Rust values.
impl<'b, 'a: 'b, T: 'b + ?Sized> From<&'b T> for TVMArgValue<'a>
where
    TVMValue: From<&'b T>,
    TVMTypeCode: From<&'b T>,
{
    fn from(arg: &'b T) -> Self {
        TVMArgValue::new(TVMValue::from(arg), TVMTypeCode::from(arg))
    }
}

/// Creates a conversion to a `TVMArgValue` for an object handle.
impl<'a, T> From<*const T> for TVMArgValue<'a> {
    fn from(ptr: *const T) -> Self {
        let value = TVMValue::new(_TVMValue {
            v_handle: ptr as *mut T as *mut c_void,
        });

        TVMArgValue::new(value, TVMTypeCode::kArrayHandle)
    }
}

/// Creates a conversion to a `TVMArgValue` for a mutable object handle.
impl<'a, T> From<*mut T> for TVMArgValue<'a> {
    fn from(ptr: *mut T) -> Self {
        let value = TVMValue::new(_TVMValue {
            v_handle: ptr as *mut c_void,
        });

        TVMArgValue::new(value, TVMTypeCode::kHandle)
    }
}

/// An owned version of TVMPODValue. It can be converted from varieties of
/// primitive and object types.
/// It can be downcasted using `try_from` if it contains the desired type.
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
    /// A primitive return value, if any.
    pub prim_value: usize,
    /// An object return value, if any.
    pub box_value: Box<Any>,
    pub type_code: TVMTypeCode,
}

impl TVMRetValue {
    fn new(prim_value: usize, box_value: Box<Any>, type_code: TVMTypeCode) -> Self {
        Self {
            prim_value,
            box_value,
            type_code,
        }
    }

    /// unsafe function to create `TVMRetValue` from `TVMValue` and
    /// its matching `TVMTypeCode`.
    pub unsafe fn from_tvm_value(value: TVMValue, type_code: TVMTypeCode) -> Self {
        let value = value.into_raw();
        match type_code {
            TVMTypeCode::kDLInt | TVMTypeCode::kDLUInt => {
                Self::new(value.v_int64 as usize, box (), type_code)
            }
            TVMTypeCode::kDLFloat => Self::new(value.v_float64 as usize, box (), type_code),
            TVMTypeCode::kHandle
            | TVMTypeCode::kArrayHandle
            | TVMTypeCode::kNodeHandle
            | TVMTypeCode::kModuleHandle
            | TVMTypeCode::kFuncHandle => {
                Self::new(value.v_handle as usize, box value.v_handle, type_code)
            }
            TVMTypeCode::kStr | TVMTypeCode::kBytes => {
                Self::new(value.v_str as usize, box (value.v_str), type_code)
            }
            _ => Self::new(0usize, box (), type_code),
        }
    }

    /// Returns the underlying `TVMValue` and `TVMTypeCode`.
    pub fn into_tvm_value(self) -> (TVMValue, TVMTypeCode) {
        let val = match self.type_code {
            TVMTypeCode::kDLInt | TVMTypeCode::kDLUInt => TVMValue::new(_TVMValue {
                v_int64: self.prim_value as i64,
            }),
            TVMTypeCode::kDLFloat => TVMValue::new(_TVMValue {
                v_float64: self.prim_value as f64,
            }),
            TVMTypeCode::kHandle
            | TVMTypeCode::kArrayHandle
            | TVMTypeCode::kNodeHandle
            | TVMTypeCode::kModuleHandle
            | TVMTypeCode::kFuncHandle
            | TVMTypeCode::kNDArrayContainer => TVMValue::new(_TVMValue {
                v_handle: self.prim_value as *const c_void as *mut c_void,
            }),
            TVMTypeCode::kStr | TVMTypeCode::kBytes => TVMValue::new(_TVMValue {
                v_str: self.prim_value as *const c_char,
            }),
            _ => unreachable!(),
        };
        (val, self.type_code)
    }
}

impl Default for TVMRetValue {
    fn default() -> Self {
        TVMRetValue {
            prim_value: 0usize,
            box_value: box (),
            type_code: TVMTypeCode::default(),
        }
    }
}

impl Clone for TVMRetValue {
    fn clone(&self) -> Self {
        match self.type_code {
            TVMTypeCode::kDLInt | TVMTypeCode::kDLUInt | TVMTypeCode::kDLFloat => {
                Self::new(self.prim_value.clone(), box (), self.type_code.clone())
            }
            TVMTypeCode::kHandle
            | TVMTypeCode::kArrayHandle
            | TVMTypeCode::kNodeHandle
            | TVMTypeCode::kModuleHandle
            | TVMTypeCode::kFuncHandle
            | TVMTypeCode::kNDArrayContainer => Self::new(
                self.prim_value.clone(),
                box (self.prim_value.clone() as *const c_void as *mut c_void),
                self.type_code.clone(),
            ),
            TVMTypeCode::kStr | TVMTypeCode::kBytes => Self::new(
                self.prim_value.clone(),
                box (self.prim_value.clone() as *const c_char),
                self.type_code.clone(),
            ),
            _ => unreachable!(),
        }
    }
}

impl Debug for TVMRetValue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "prim_value: {:?}, box_value: {:?}, type_code: {:?}",
            self.prim_value, self.prim_value as *const c_void as *mut c_void, self.type_code
        )
    }
}

macro_rules! impl_prim_ret_value {
    ($type:ty, $code:expr) => {
        impl From<$type> for TVMRetValue {
            fn from(val: $type) -> Self {
                TVMRetValue {
                    prim_value: val as usize,
                    box_value: box (),
                    type_code: $code,
                }
            }
        }

        impl<'a> From<&'a $type> for TVMRetValue {
            fn from(val: &$type) -> Self {
                TVMRetValue {
                    prim_value: *val as usize,
                    box_value: box (),
                    type_code: $code,
                }
            }
        }

        impl<'a> From<&'a mut $type> for TVMRetValue {
            fn from(val: &mut $type) -> Self {
                TVMRetValue {
                    prim_value: *val as usize,
                    box_value: box (),
                    type_code: $code,
                }
            }
        }

        impl TryFrom<TVMRetValue> for $type {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$type> {
                if ret.type_code == $code {
                    Ok(ret.prim_value as $type)
                } else {
                    bail!(ErrorKind::TryFromTVMRetValueError(
                        stringify!($type).to_string(),
                        ret.type_code.to_string(),
                    ))
                }
            }
        }
    };
}

impl_prim_ret_value!(i8, TVMTypeCode::kDLInt);
impl_prim_ret_value!(i16, TVMTypeCode::kDLInt);
impl_prim_ret_value!(i32, TVMTypeCode::kDLInt);
impl_prim_ret_value!(i64, TVMTypeCode::kDLInt);
impl_prim_ret_value!(isize, TVMTypeCode::kDLInt);

impl_prim_ret_value!(u8, TVMTypeCode::kDLUInt);
impl_prim_ret_value!(u16, TVMTypeCode::kDLUInt);
impl_prim_ret_value!(u32, TVMTypeCode::kDLUInt);
impl_prim_ret_value!(u64, TVMTypeCode::kDLUInt);
impl_prim_ret_value!(usize, TVMTypeCode::kDLUInt);

impl_prim_ret_value!(f32, TVMTypeCode::kDLFloat);
impl_prim_ret_value!(f64, TVMTypeCode::kDLFloat);

macro_rules! impl_ptr_ret_value {
    ($type:ty) => {
        impl From<$type> for TVMRetValue {
            fn from(ptr: $type) -> Self {
                TVMRetValue {
                    prim_value: ptr as usize,
                    box_value: box (),
                    type_code: TVMTypeCode::kHandle,
                }
            }
        }

        impl TryFrom<TVMRetValue> for $type {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$type> {
                if ret.type_code == TVMTypeCode::kHandle {
                    Ok(ret.prim_value as $type)
                } else {
                    bail!(ErrorKind::TryFromTVMRetValueError(
                        stringify!($type).to_string(),
                        ret.type_code.to_string(),
                    ))
                }
            }
        }
    };
}

impl_ptr_ret_value!(*const c_void);
impl_ptr_ret_value!(*mut c_void);

impl From<String> for TVMRetValue {
    fn from(val: String) -> Self {
        let pval = val.as_ptr() as *const c_char as usize;
        let bval = box (val.as_ptr() as *const c_char);
        mem::forget(val);
        TVMRetValue::new(pval, bval, TVMTypeCode::kStr)
    }
}

impl TryFrom<TVMRetValue> for String {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<String> {
        // Note: simple downcast doesn't work for function call return values
        let ret_str = unsafe {
            match CStr::from_ptr(ret.prim_value as *const c_char).to_str() {
                Ok(s) => s,
                Err(_) => "Invalid UTF-8 message",
            }
        };

        Ok(ret_str.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn numeric() {
        macro_rules! arg_ret_tests {
            ($v:expr; $($ty:ty),+) => {{
                $(
                    let v = $v as $ty;
                    let b = TVMRetValue::from(&v);
                    let b: $ty = b.try_into().unwrap();
                    assert_eq!(b, v);
                )+
            }};
        }

        arg_ret_tests!(42; i8, i16, i32, i64, f32, f64);
    }

    #[test]
    fn string() {
        let s = "hello".to_string();
        let tvm_arg: String = TVMRetValue::from(s.clone()).try_into().unwrap();
        assert_eq!(tvm_arg, s);
    }
}
