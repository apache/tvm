//! This module implements [`TVMArgValue`] and [`TVMRetValue`] types
//! and their conversions needed for the types used in frontend crate.
//! `TVMRetValue` is the owned version of `TVMPODValue`.

use std::{convert::TryFrom, os::raw::c_void};

use failure::Error;
use tvm_common::{
    ensure_type,
    ffi::{self, TVMValue},
};

use crate::{
    common_errors::*, context::TVMContext, Function, Module, NDArray, TVMArgValue, TVMByteArray,
    TVMRetValue,
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
            fn try_from(arg: &TVMArgValue<'v>) -> Result<$ty, Self::Error> {
                ensure_type!(arg, $type_code);
                Ok($ty::new(unsafe { arg.value.v_handle as $handle }))
            }
        }

        impl From<$ty> for TVMRetValue {
            fn from(val: $ty) -> TVMRetValue {
                TVMRetValue {
                    value: TVMValue {
                        v_handle: val.handle() as *mut c_void,
                    },
                    box_value: box val,
                    type_code: $type_code as i64,
                }
            }
        }

        impl TryFrom<TVMRetValue> for $ty {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$ty, Self::Error> {
                ensure_type!(ret, $type_code);
                Ok($ty::new(unsafe { ret.value.v_handle as $handle }))
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

macro_rules! impl_boxed_ret_value {
    ($type:ty, $code:expr) => {
        impl From<$type> for TVMRetValue {
            fn from(val: $type) -> Self {
                TVMRetValue {
                    value: TVMValue { v_int64: 0 },
                    box_value: box val,
                    type_code: $code as i64,
                }
            }
        }
        impl TryFrom<TVMRetValue> for $type {
            type Error = Error;
            fn try_from(ret: TVMRetValue) -> Result<$type, Self::Error> {
                if let Ok(val) = ret.box_value.downcast::<$type>() {
                    Ok(*val)
                } else {
                    bail!(ValueDowncastError::new($code as i64, ret.type_code as i64))
                }
            }
        }
    };
}

impl_boxed_ret_value!(TVMContext, ffi::TVMTypeCode_kTVMContext);
impl_boxed_ret_value!(TVMByteArray, ffi::TVMTypeCode_kBytes);

impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for TVMByteArray {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'v>) -> Result<Self, Self::Error> {
        ensure_type!(arg, ffi::TVMTypeCode_kBytes);
        Ok(TVMByteArray::new(unsafe {
            *(arg.value.v_handle as *mut ffi::TVMByteArray)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{convert::TryInto, str::FromStr};
    use tvm_common::ffi::TVMType;

    #[test]
    fn bytearray() {
        let w = vec![1u8, 2, 3, 4, 5];
        let v = TVMByteArray::from(&w);
        let tvm: TVMByteArray = TVMRetValue::from(v).try_into().unwrap();
        assert_eq!(tvm.data(), w.iter().map(|e| *e as i8).collect::<Vec<i8>>());
    }

    #[test]
    fn ty() {
        let t = TVMType::from_str("int32").unwrap();
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
