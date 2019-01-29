use std::{convert::TryFrom, marker::PhantomData, os::raw::c_void};

use super::Tensor;
use crate::ffi::runtime::{
    BackendPackedCFunc, DLTensor as _DLTensor, TVMTypeCode_kArrayHandle,
    TVMTypeCode_kNDArrayContainer, TVMValue as _TVMValue,
};

use super::DLTensor;
use crate::{
    common::{TVMArgValue, TVMRetValue, TVMTypeCode, TVMValue},
    errors::*,
};

pub type PackedFunc = Box<Fn(&[TVMArgValue]) -> TVMRetValue + Send + Sync>;

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

impl<'a> From<&'a DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a DLTensor) -> Self {
        let raw = _TVMValue {
            v_handle: arr as *const _ as *mut DLTensor as *mut c_void,
        };
        TVMArgValue {
            value: TVMValue::new(raw),
            type_code: TVMTypeCode::kArrayHandle,
            lifetime: PhantomData,
        }
    }
}

impl<'a> From<&'a mut DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a mut DLTensor) -> Self {
        let raw = _TVMValue {
            v_handle: arr as *mut _ as *mut c_void,
        };
        TVMArgValue {
            value: TVMValue::new(raw),
            type_code: TVMTypeCode::kArrayHandle,
            lifetime: PhantomData,
        }
    }
}

impl<'a> TryFrom<TVMArgValue<'a>> for Tensor<'a> {
    type Error = Error;
    fn try_from(val: TVMArgValue<'a>) -> Result<Self> {
        ensure!(
            val.type_code == TVMTypeCode::kArrayHandle
                || val.type_code == TVMTypeCode::kNDArrayContainer,
            "Could not downcast arg. Expected `{}` or `{}`, but got `{}`",
            TVMTypeCode::kArrayHandle,
            TVMTypeCode::kNDArrayContainer,
            val.type_code,
        );

        let dlt = unsafe { *(val.value.v_handle as *mut _DLTensor as *const _DLTensor) };
        Ok(DLTensor::new(dlt).into())
    }
}

impl<'a, 't> From<&'t Tensor<'a>> for TVMRetValue {
    fn from(val: &'t Tensor<'a>) -> Self {
        TVMRetValue {
            prim_value: 0,
            box_value: box DLTensor::from(val),
            type_code: TVMTypeCode::kNDArrayContainer,
        }
    }
}

impl<'a> TryFrom<TVMRetValue> for Tensor<'a> {
    type Error = Error;
    fn try_from(ret: TVMRetValue) -> Result<Self> {
        ensure!(
            ret.type_code == TVMTypeCode::kArrayHandle
                || ret.type_code == TVMTypeCode::kNDArrayContainer,
            "Could not downcast arg. Expected `{}` or `{}`, but got `{}`",
            TVMTypeCode_kArrayHandle,
            TVMTypeCode_kNDArrayContainer,
            ret.type_code,
        );

        let dlt = unsafe { *(ret.prim_value as *mut _DLTensor as *const _DLTensor) };
        Ok(DLTensor::new(dlt).into())
    }
}

// @see `WrapPackedFunc` in `llvm_module.cc`.
pub(crate) fn wrap_backend_packed_func(func: BackendPackedCFunc) -> PackedFunc {
    box move |args: &[TVMArgValue]| {
        func(
            args.iter()
                .map(|ref arg| arg.value.inner)
                .collect::<Vec<_TVMValue>>()
                .as_ptr(),
            args.iter()
                .map(|ref arg| arg.type_code as i32)
                .collect::<Vec<i32>>()
                .as_ptr() as *const i32,
            args.len() as i32,
        );
        TVMRetValue::default()
    }
}
