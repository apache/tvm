use std::{any::Any, convert::TryFrom, marker::PhantomData, os::raw::c_void};

use ffi::runtime::{
  BackendPackedCFunc, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLTensor,
  TVMTypeCode_kArrayHandle, TVMTypeCode_kHandle, TVMValue,
};

use errors::*;

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

/// A borrowed TVMPODValue. Can be constructed using `into()` but the preferred way
/// to obtain a `TVMArgValue` is automatically via `call_packed!`.
#[derive(Clone, Copy)]
pub struct TVMArgValue<'a> {
  _lifetime: PhantomData<&'a ()>,
  pub(crate) value: TVMValue,
  pub(crate) type_code: i64,
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

/// Creates a conversion to a `TVMArgValue` for a primitive type and DLDataTypeCode.
macro_rules! impl_prim_tvm_arg {
  ($type:ty, $field:ident, $code:expr, $as:ty) => {
    impl<'a> From<$type> for TVMArgValue<'a> {
      fn from(val: $type) -> Self {
        TVMArgValue {
          value: TVMValue { $field: val as $as },
          type_code: $code as i64,
          _lifetime: PhantomData,
        }
      }
    }
  };
  ($type:ty, $field:ident, $code:expr) => {
    impl_prim_tvm_arg!($type, $field, $code, $type);
  };
  ($type:ty,v_int64) => {
    impl_prim_tvm_arg!($type, v_int64, DLDataTypeCode_kDLInt, i64);
  };
  ($type:ty,v_float64) => {
    impl_prim_tvm_arg!($type, v_float64, DLDataTypeCode_kDLFloat, f64);
  };
}

impl_prim_tvm_arg!(f32, v_float64);
impl_prim_tvm_arg!(f64, v_float64);
impl_prim_tvm_arg!(i8, v_int64);
impl_prim_tvm_arg!(u8, v_int64);
impl_prim_tvm_arg!(i32, v_int64);
impl_prim_tvm_arg!(u32, v_int64);
impl_prim_tvm_arg!(i64, v_int64);
impl_prim_tvm_arg!(u64, v_int64);
impl_prim_tvm_arg!(bool, v_int64);

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
  /// A primitive return value, if any.
  prim_value: u64,
  /// An object return value, if any.
  box_value: Box<Any>,
  /// The DLDataTypeCode which determines whether `prim_value` or `box_value` is in use.
  type_code: i64,
}

#[cfg(target_env = "sgx")]
impl TVMRetValue {
  pub(crate) fn from_tvm_value(value: TVMValue, type_code: i64) -> Self {
    unsafe {
      Self {
        prim_value: match type_code {
          0 | 1 => value.v_int64 as u64,
          2 => value.v_float64 as u64,
          3 | 7 | 8 | 9 | 10 => value.v_handle as u64,
          11 | 12 => value.v_str as u64,
          _ => 0,
        } as u64,
        box_value: box (),
        type_code: type_code,
      }
    }
  }

  pub fn into_tvm_value(self) -> (TVMValue, i64) {
    let val = match self.type_code {
      0 | 1 => TVMValue {
        v_int64: self.prim_value.clone() as i64,
      },
      2 => TVMValue {
        v_float64: self.prim_value.clone() as f64,
      },
      3 | 7 | 8 | 9 | 10 => TVMValue {
        v_handle: Box::into_raw(self.box_value) as *mut c_void,
      },
      11 | 12 => TVMValue {
        v_str: Box::into_raw(self.box_value) as *const _,
      },
      _ => unreachable!(),
    };
    (val, self.type_code)
  }
}

impl Default for TVMRetValue {
  fn default() -> Self {
    TVMRetValue {
      prim_value: 0,
      box_value: box (),
      type_code: 0,
    }
  }
}

macro_rules! impl_prim_ret_value {
  ($type:ty, $code:expr) => {
    impl From<$type> for TVMRetValue {
      fn from(val: $type) -> Self {
        TVMRetValue {
          prim_value: val as u64,
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
            ret.type_code
          ))
        }
      }
    }
  };
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
            ret.type_code
          ))
        }
      }
    }
  };
}

impl_prim_ret_value!(i8, 0);
impl_prim_ret_value!(u8, 1);
impl_prim_ret_value!(i16, 0);
impl_prim_ret_value!(u16, 1);
impl_prim_ret_value!(i32, 0);
impl_prim_ret_value!(u32, 1);
impl_prim_ret_value!(f32, 2);
impl_prim_ret_value!(i64, 0);
impl_prim_ret_value!(u64, 1);
impl_prim_ret_value!(f64, 2);
impl_prim_ret_value!(isize, 0);
impl_prim_ret_value!(usize, 1);
impl_boxed_ret_value!(String, 11);

// @see `WrapPackedFunc` in `llvm_module.cc`.
pub(super) fn wrap_backend_packed_func(func: BackendPackedCFunc) -> PackedFunc {
  box move |args: &[TVMArgValue]| {
    func(
      args
        .iter()
        .map(|ref arg| arg.value)
        .collect::<Vec<TVMValue>>()
        .as_ptr(),
      args
        .iter()
        .map(|ref arg| arg.type_code as i32)
        .collect::<Vec<i32>>()
        .as_ptr() as *const i32,
      args.len() as i32,
    );
    TVMRetValue::default()
  }
}
