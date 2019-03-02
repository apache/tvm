//! This module implements [`TVMArgValue`] and [`TVMRetValue`] types
//! and their conversions needed for the types used in frontend crate.
//! `TVMRetValue` is the owned version of `TVMPODValue`.

use std::{convert::TryFrom, mem, os::raw::c_void};

use crate::{
    common_errors::*, ts, Function, Module, NDArray, TVMArgValue, TVMByteArray, TVMContext,
    TVMDeviceType, TVMRetValue, TVMType, TVMTypeCode, TVMValue,
};

macro_rules! impl_tvm_val_from_handle {
    ($($ty:ty),+) => {
        $(
            impl<'a> From<&'a $ty> for TVMValue {
                fn from(arg: &$ty) -> Self {
                    let inner = ts::TVMValue {
                        v_handle: arg.handle as *mut _ as *mut c_void,
                    };
                    Self::new(inner)
                }
            }
        )+
    }
}

impl_tvm_val_from_handle!(Module, Function, NDArray);

impl<'a> From<&'a TVMType> for TVMValue {
    fn from(ty: &TVMType) -> Self {
        let inner = ts::TVMValue { v_type: ty.inner };
        Self::new(inner)
    }
}

impl<'a> From<&'a TVMContext> for TVMValue {
    fn from(ctx: &TVMContext) -> Self {
        let inner = ts::TVMValue {
            v_ctx: ctx.clone().into(),
        };
        Self::new(inner)
    }
}

impl<'a> From<&'a TVMDeviceType> for TVMValue {
    fn from(dev: &TVMDeviceType) -> Self {
        let inner = ts::TVMValue {
            v_int64: dev.0 as i64,
        };
        Self::new(inner)
    }
}

impl<'a> From<&'a TVMByteArray> for TVMValue {
    fn from(barr: &TVMByteArray) -> Self {
        let inner = ts::TVMValue {
            v_handle: &barr.inner as *const ts::TVMByteArray as *mut c_void,
        };
        Self::new(inner)
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for NDArray {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kArrayHandle {
            let handle = unsafe { arg.value.inner.v_handle };
            let arr_handle = unsafe { mem::transmute::<*mut c_void, ts::TVMArrayHandle>(handle) };
            Ok(Self::new(arr_handle, true))
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(NDArray).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for Module {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kModuleHandle {
            let handle = unsafe { arg.value.inner.v_handle };
            Ok(Self::new(handle, false))
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(Module).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for TVMByteArray {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kBytes {
            unsafe {
                let barr_ptr =
                    mem::transmute::<*mut c_void, *mut ts::TVMByteArray>(arg.value.inner.v_handle);
                Ok(Self::new(*barr_ptr))
            }
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(TVMByteArray).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for TVMType {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kTVMType {
            let ty = unsafe { arg.value.inner.v_type };
            Ok(TVMType::from(ty))
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(TVMType).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

impl<'a, 'b> TryFrom<&'b TVMArgValue<'a>> for TVMContext {
    type Error = Error;
    fn try_from(arg: &TVMArgValue<'a>) -> Result<Self> {
        if arg.type_code == TVMTypeCode::kTVMContext {
            let ty = unsafe { arg.value.inner.v_ctx };
            Ok(TVMContext::from(ty))
        } else {
            bail!(ErrorKind::TryFromTVMArgValueError(
                stringify!(TVMContext).to_string(),
                arg.type_code.to_string()
            ))
        }
    }
}

macro_rules! impl_boxed_ret_value {
    ($type:ty, $code:expr) => {
        impl From<$type> for TVMRetValue {
            fn from(val: $type) -> Self {
                TVMRetValue {
                    prim_value: 0,
                    box_value: box val,
                    type_code: $code,
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
                        ret.type_code.to_string()
                    ))
                }
            }
        }
    };
}

impl_boxed_ret_value!(TVMType, TVMTypeCode::kTVMType);
impl_boxed_ret_value!(TVMContext, TVMTypeCode::kTVMContext);
impl_boxed_ret_value!(TVMByteArray, TVMTypeCode::kBytes);

impl TryFrom<TVMRetValue> for Module {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<Module> {
        if let Ok(handle) = ret.box_value.downcast::<ts::TVMModuleHandle>() {
            Ok(Module::new(*handle, false))
        } else {
            bail!(ErrorKind::TryFromTVMRetValueError(
                stringify!(TVMTypeCode::kModuleHandle).to_string(),
                ret.type_code.to_string()
            ))
        }
    }
}

impl TryFrom<TVMRetValue> for Function {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<Function> {
        if let Ok(handle) = ret.box_value.downcast::<ts::TVMFunctionHandle>() {
            Ok(Function::new(*handle, false, false))
        } else {
            bail!(ErrorKind::TryFromTVMRetValueError(
                stringify!(TVMTypeCode::kFuncHandle).to_string(),
                ret.type_code.to_string()
            ))
        }
    }
}

impl TryFrom<TVMRetValue> for NDArray {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<NDArray> {
        if let Ok(handle) = ret.box_value.downcast::<ts::TVMArrayHandle>() {
            Ok(NDArray::new(*handle, false))
        } else {
            bail!(ErrorKind::TryFromTVMRetValueError(
                stringify!(TVMTypeCode::kArrayHandle).to_string(),
                ret.type_code.to_string()
            ))
        }
    }
}

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
