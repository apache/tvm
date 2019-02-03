//! This module implements [`TVMArgValue`] and [`TVMRetValue`] types
//! and their conversions needed for the types used in frontend crate.
//! `TVMRetValue` is the owned version of `TVMPODValue`.

use std::{convert::TryFrom, os::raw::c_void};

use tvm_common::{
    ensure_type,
    ffi::{self, TVMValue},
};

use crate::{
    common_errors::*, Function, Module, NDArray, TVMArgValue, TVMByteArray, TVMContext, TVMRetValue,
};

macro_rules! impl_tvm_val_from_handle {
    ($ty:ident, $type_code:expr, $handle:ty) => {
        impl<'a> From<&'a $ty> for TVMArgValue<'a> {
            fn from(arg: &$ty) -> Self {
                TVMArgValue {
                    value: TVMValue {
                        v_handle: arg.handle as *mut _ as *mut c_void,
                    },
                    type_code: $type_code as i64,
                    _lifetime: std::marker::PhantomData,
                }
            }
        }

        impl<'a> From<&'a mut $ty> for TVMArgValue<'a> {
            fn from(arg: &mut $ty) -> Self {
                TVMArgValue {
                    value: TVMValue {
                        v_handle: arg.handle as *mut _ as *mut c_void,
                    },
                    type_code: $type_code as i64,
                    _lifetime: std::marker::PhantomData,
                }
            }
        }

        impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for $ty {
            type Error = Error;
            fn try_from(arg: &TVMArgValue<'v>) -> Result<$ty> {
                ensure_type!(arg, $type_code);
                Ok($ty::new(unsafe { *(arg.value.v_handle as *const $handle) }))
            }
        }

        impl From<$ty> for TVMRetValue {
            fn from(val: $ty) -> TVMRetValue {
                TVMRetValue {
                    prim_value: 0,
                    box_value: box val,
                    type_code: $type_code as i64,
                }
            }
        }

        impl TryFrom<TVMRetValue> for $ty {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$ty> {
                if let Ok(handle) = ret.box_value.downcast::<$handle>() {
                    Ok($ty::new(*handle))
                } else {
                    bail!(ErrorKind::TryFromTVMRetValueError(
                        stringify!($type_code).to_string(),
                        ret.type_code,
                    ))
                }
            }
        }
    };
}

impl_tvm_val_from_handle!(
    Function,
    ffi::TVMTypeCode_kFuncHandle,
    ffi::TVMFunctionHandle
);
impl_tvm_val_from_handle!(Module, ffi::TVMTypeCode_kModuleHandle, ffi::TVMModuleHandle);
impl_tvm_val_from_handle!(NDArray, ffi::TVMTypeCode_kArrayHandle, ffi::TVMArrayHandle);

impl<'a> From<&'a TVMByteArray> for TVMValue {
    fn from(barr: &TVMByteArray) -> Self {
        TVMValue {
            v_handle: &barr.inner as *const ffi::TVMByteArray as *mut c_void,
        }
    }
}

impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for TVMByteArray {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'v>) -> Result<Self> {
        ensure_type!(arg, ffi::TVMTypeCode_kBytes);
        Ok(TVMByteArray::new(unsafe {
            *(arg.value.v_handle as *mut ffi::TVMByteArray)
        }))
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for TVMContext {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        ensure_type!(arg, ffi::TVMTypeCode_kTVMContext);
        Ok(unsafe { arg.value.v_ctx.into() })
    }
}

macro_rules! impl_boxed_ret_value {
    ($type:ty, $code:expr) => {
        impl From<$type> for TVMRetValue {
            fn from(val: $type) -> Self {
                TVMRetValue {
                    prim_value: 0,
                    box_value: box val,
                    type_code: $code as i64,
                }
            }
        }
        impl TryFrom<TVMRetValue> for $type {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$type> {
                if let Ok(val) = ret.box_value.downcast::<$type>() {
                    Ok(*val)
                } else {
                    bail!(ErrorKind::TryFromTVMRetValueError(
                        stringify!($type).to_string(),
                        ret.type_code
                    ))
                }
            }
        }
    };
}

// impl_boxed_ret_value!(TVMType, ffi::TVMTypeCode_kTVMType);
impl_boxed_ret_value!(TVMContext, ffi::TVMTypeCode_kTVMContext);
impl_boxed_ret_value!(TVMByteArray, ffi::TVMTypeCode_kBytes);

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;

    #[test]
    fn bytearray() {
        let w = vec![1u8, 2, 3, 4, 5];
        let v = TVMByteArray::from(&w);
        let tvm: TVMByteArray = TVMRetValue::from(v).try_into().unwrap();
        assert_eq!(tvm.data(), w.iter().map(|e| *e as i8).collect::<Vec<i8>>());
    }

    #[test]
    fn ty() {
        let t = TVMType::from("int32");
        let tvm: TVMType = TVMRetValue::from(t).try_into().unwrap();
        assert_eq!(tvm, t);
    }

    #[test]
    fn ctx() {
        let c = TVMContext::from("gpu");
        let tvm: TVMContext = TVMRetValue::from(c).try_into().unwrap();
        assert_eq!(tvm, c);
    }
}
