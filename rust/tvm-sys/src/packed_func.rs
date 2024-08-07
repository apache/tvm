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

use std::{
    convert::TryFrom,
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
};

use crate::{errors::ValueDowncastError, ffi::*};

pub use crate::ffi::TVMValue;

pub trait PackedFunc:
    Fn(&[ArgValue]) -> Result<RetValue, crate::errors::FuncCallError> + Send + Sync
{
}

impl<T> PackedFunc for T where
    T: Fn(&[ArgValue]) -> Result<RetValue, crate::errors::FuncCallError> + Send + Sync
{
}

/// Calls a packed function and returns a `RetValue`.
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

/// Constructs a derivative of a TVMPodValue.
macro_rules! TVMPODValue {
    {
        $(#[$m:meta])+
        $name:ident $(<$a:lifetime>)? {
            $($extra_variant:ident ( $variant_type:ty ) ),+ $(,)?
        },
        match $value:ident {
            $($tvm_type:ident => { $from_tvm_type:expr })+
        },
        match &self {
            $($self_type:ident ( $val:ident ) => { $from_self_type:expr })+
        }
        $(,)?
    } => {
        $(#[$m])+
        #[derive(Clone, Debug)]
        pub enum $name $(<$a>)? {
            Int(i64),
            UInt(i64),
            Float(f64),
            Bool(bool),
            Null,
            DataType(DLDataType),
            String(*mut c_char),
            Device(DLDevice),
            Handle(*mut c_void),
            ArrayHandle(TVMArrayHandle),
            ObjectHandle(*mut c_void),
            ModuleHandle(TVMModuleHandle),
            FuncHandle(TVMFunctionHandle),
            NDArrayHandle(*mut c_void),
            $($extra_variant($variant_type)),+
        }

        impl $(<$a>)? $name $(<$a>)? {
            pub fn from_tvm_value($value: TVMValue, type_code: u32) -> Self {
                use $name::*;
                #[allow(non_upper_case_globals)]
                unsafe {
                    match type_code as _ {
                        DLDataTypeCode_kDLInt => Int($value.v_int64),
                        DLDataTypeCode_kDLUInt => UInt($value.v_int64),
                        DLDataTypeCode_kDLFloat => Float($value.v_float64),
                        TVMArgTypeCode_kTVMArgBool => Bool($value.v_int64 != 0),
                        TVMArgTypeCode_kTVMNullptr => Null,
                        TVMArgTypeCode_kTVMDataType => DataType($value.v_type),
                        TVMArgTypeCode_kDLDevice => Device($value.v_device),
                        TVMArgTypeCode_kTVMOpaqueHandle => Handle($value.v_handle),
                        TVMArgTypeCode_kTVMDLTensorHandle => ArrayHandle($value.v_handle as TVMArrayHandle),
                        TVMArgTypeCode_kTVMObjectHandle => ObjectHandle($value.v_handle),
                        TVMArgTypeCode_kTVMObjectRValueRefArg => ObjectHandle(*($value.v_handle as *mut *mut c_void)),
                        TVMArgTypeCode_kTVMModuleHandle => ModuleHandle($value.v_handle),
                        TVMArgTypeCode_kTVMPackedFuncHandle => FuncHandle($value.v_handle),
                        TVMArgTypeCode_kTVMNDArrayHandle => NDArrayHandle($value.v_handle),
                        $( $tvm_type => { $from_tvm_type } ),+
                        _ => unimplemented!("{}", type_code),
                    }
                }
            }

            pub fn to_tvm_value(&self) -> (TVMValue, TVMArgTypeCode) {
                use $name::*;
                match self {
                    Int(val) => (TVMValue { v_int64: *val }, DLDataTypeCode_kDLInt),
                    UInt(val) => (TVMValue { v_int64: *val as i64 }, DLDataTypeCode_kDLUInt),
                    Float(val) => (TVMValue { v_float64: *val }, DLDataTypeCode_kDLFloat),
                    Bool(val) => (TVMValue { v_int64: *val as i64 }, TVMArgTypeCode_kTVMArgBool),
                    Null => (TVMValue{ v_int64: 0 },TVMArgTypeCode_kTVMNullptr),
                    DataType(val) => (TVMValue { v_type: *val }, TVMArgTypeCode_kTVMDataType),
                    Device(val) => (TVMValue { v_device: val.clone() }, TVMArgTypeCode_kDLDevice),
                    String(val) => {
                        (
                            TVMValue { v_handle: *val as *mut c_void },
                            TVMArgTypeCode_kTVMStr,
                        )
                    }
                    Handle(val) => (TVMValue { v_handle: *val }, TVMArgTypeCode_kTVMOpaqueHandle),
                    ArrayHandle(val) => {
                        (
                            TVMValue { v_handle: *val as *const _ as *mut c_void },
                            TVMArgTypeCode_kTVMNDArrayHandle,
                        )
                    },
                    ObjectHandle(val) => (TVMValue { v_handle: *val }, TVMArgTypeCode_kTVMObjectHandle),
                    ModuleHandle(val) =>
                        (TVMValue { v_handle: *val }, TVMArgTypeCode_kTVMModuleHandle),
                    FuncHandle(val) => (
                        TVMValue { v_handle: *val },
                        TVMArgTypeCode_kTVMPackedFuncHandle
                    ),
                    NDArrayHandle(val) =>
                        (TVMValue { v_handle: *val }, TVMArgTypeCode_kTVMNDArrayHandle),
                    $( $self_type($val) => { $from_self_type } ),+
                }
            }
        }
    }
}

TVMPODValue! {
    /// A borrowed TVMPODValue. Can be constructed using `into()` but the preferred way
    /// to obtain a `ArgValue` is automatically via `call_packed!`.
    ArgValue<'a> {
        Bytes(&'a TVMByteArray),
        Str(&'a CStr),
    },
    match value {
        TVMArgTypeCode_kTVMBytes => { Bytes(&*(value.v_handle as *const TVMByteArray)) }
        TVMArgTypeCode_kTVMStr => { Str(CStr::from_ptr(value.v_handle as *const i8)) }
    },
    match &self {
        Bytes(val) => {
            (TVMValue { v_handle: *val as *const _ as *mut c_void }, TVMArgTypeCode_kTVMBytes)
        }
        Str(val) => { (TVMValue { v_handle: val.as_ptr() as *mut c_void }, TVMArgTypeCode_kTVMStr) }
    }
}

TVMPODValue! {
    /// An owned TVMPODValue. Can be converted from a variety of primitive and object types.
    /// Can be downcasted using `try_from` if it contains the desired type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::convert::{TryFrom, TryInto};
    /// use tvm_sys::RetValue;
    ///
    /// let a = 42u32;
    /// let b: u32 = tvm_sys::RetValue::from(a).try_into().unwrap();
    ///
    /// let s = "hello, world!";
    /// let t: RetValue = s.to_string().into();
    /// assert_eq!(String::try_from(t).unwrap(), s);
    /// ```
    RetValue {
        Bytes(TVMByteArray),
        Str(&'static CStr),
    },
    match value {
        TVMArgTypeCode_kTVMBytes => { Bytes(*(value.v_handle as *const TVMByteArray)) }
        TVMArgTypeCode_kTVMStr => { Str(CStr::from_ptr(value.v_handle as *mut i8)) }
    },
    match &self {
        Bytes(val) =>
            { (TVMValue { v_handle: val as *const _ as *mut c_void }, TVMArgTypeCode_kTVMBytes ) }
        Str(val) =>
            { (TVMValue { v_str: val.as_ptr() }, TVMArgTypeCode_kTVMStr ) }
    }
}

#[macro_export]
macro_rules! try_downcast {
    ($val:ident -> $into:ty, $( |$pat:pat| { $converter:expr } ),+ ) => {
        match $val {
            $( $pat => { Ok($converter) } )+
            _ => Err($crate::errors::ValueDowncastError {
                actual_type: format!("{:?}", $val),
                expected_type: stringify!($into),
            }),
        }
    };
}

/// Creates a conversion to a `ArgValue` for a primitive type and DLDataTypeCode.
macro_rules! impl_pod_value {
    ($variant:ident, $inner_ty:ty, [ $( $type:ty ),+ ] ) => {
        $(
            impl<'a> From<$type> for ArgValue<'a> {
                fn from(val: $type) -> Self {
                    Self::$variant(val as $inner_ty)
                }
            }

            impl<'a> From<&'a $type> for ArgValue<'a> {
                fn from(val: &'a $type) -> Self {
                    Self::$variant(*val as $inner_ty)
                }
            }

            impl<'a> TryFrom<ArgValue<'a>> for $type {
                type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: ArgValue<'a>) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |ArgValue::$variant(val)| { val as $type })
                }
            }

            impl<'a, 'v> TryFrom<&'a ArgValue<'v>> for $type {
                type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: &'a ArgValue<'v>) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |ArgValue::$variant(val)| { *val as $type })
                }
            }

            impl From<$type> for RetValue {
                fn from(val: $type) -> Self {
                    Self::$variant(val as $inner_ty)
                }
            }

            impl TryFrom<RetValue> for $type {
              type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: RetValue) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |RetValue::$variant(val)| { val as $type })
                }
            }
        )+
    };
}

impl_pod_value!(Int, i64, [i8, i16, i32, i64, isize]);
impl_pod_value!(UInt, i64, [u8, u16, u32, u64, usize]);
impl_pod_value!(Float, f64, [f32, f64]);
impl_pod_value!(Bool, bool, [bool]);
impl_pod_value!(DataType, DLDataType, [DLDataType]);
impl_pod_value!(Device, DLDevice, [DLDevice]);

impl<'a> From<&'a str> for ArgValue<'a> {
    fn from(s: &'a str) -> Self {
        Self::String(CString::new(s).unwrap().into_raw())
    }
}

impl<'a> From<String> for ArgValue<'a> {
    fn from(s: String) -> Self {
        Self::String(CString::new(s).unwrap().into_raw())
    }
}

impl<'a> From<&'a CStr> for ArgValue<'a> {
    fn from(s: &'a CStr) -> Self {
        Self::Str(s)
    }
}

impl<'a> From<&'a CString> for ArgValue<'a> {
    fn from(s: &'a CString) -> Self {
        Self::String(s.as_ptr() as _)
    }
}

impl<'a> From<&'a TVMByteArray> for ArgValue<'a> {
    fn from(s: &'a TVMByteArray) -> Self {
        Self::Bytes(s)
    }
}

impl<'a> TryFrom<ArgValue<'a>> for &'a str {
    type Error = ValueDowncastError;
    fn try_from(val: ArgValue<'a>) -> Result<Self, Self::Error> {
        try_downcast!(val -> &str, |ArgValue::Str(s)| { s.to_str().unwrap() })
    }
}

impl<'a, 'v> TryFrom<&'a ArgValue<'v>> for &'v str {
    type Error = ValueDowncastError;
    fn try_from(val: &'a ArgValue<'v>) -> Result<Self, Self::Error> {
        try_downcast!(val -> &str, |ArgValue::Str(s)| { s.to_str().unwrap() })
    }
}

/// Converts an unspecialized handle to a ArgValue.
impl<'a, T> From<*const T> for ArgValue<'a> {
    fn from(ptr: *const T) -> Self {
        Self::Handle(ptr as *mut c_void)
    }
}

/// Converts an unspecialized mutable handle to a ArgValue.
impl<'a, T> From<*mut T> for ArgValue<'a> {
    fn from(ptr: *mut T) -> Self {
        Self::Handle(ptr as *mut c_void)
    }
}

impl<'a> From<&'a mut DLTensor> for ArgValue<'a> {
    fn from(arr: &'a mut DLTensor) -> Self {
        Self::ArrayHandle(arr as *mut DLTensor)
    }
}

impl<'a> From<&'a DLTensor> for ArgValue<'a> {
    fn from(arr: &'a DLTensor) -> Self {
        Self::ArrayHandle(arr as *const _ as *mut DLTensor)
    }
}

impl TryFrom<RetValue> for String {
    type Error = ValueDowncastError;
    fn try_from(val: RetValue) -> Result<String, Self::Error> {
        try_downcast!(
            val -> String,
            |RetValue::String(s)| { unsafe { CString::from_raw(s).into_string().unwrap() }},
            |RetValue::Str(s)| { s.to_str().unwrap().to_string() }
        )
    }
}

impl From<String> for RetValue {
    fn from(s: String) -> Self {
        Self::String(std::ffi::CString::new(s).unwrap().into_raw())
    }
}

impl From<TVMByteArray> for RetValue {
    fn from(arr: TVMByteArray) -> Self {
        Self::Bytes(arr)
    }
}

impl TryFrom<RetValue> for TVMByteArray {
    type Error = ValueDowncastError;
    fn try_from(val: RetValue) -> Result<Self, Self::Error> {
        try_downcast!(val -> TVMByteArray, |RetValue::Bytes(val)| { val })
    }
}

impl Default for RetValue {
    fn default() -> Self {
        Self::Int(0)
    }
}

impl TryFrom<RetValue> for std::ffi::CString {
    type Error = ValueDowncastError;
    fn try_from(val: RetValue) -> Result<CString, Self::Error> {
        try_downcast!(val -> std::ffi::CString,
            |RetValue::Str(val)| { val.into() })
    }
}

impl From<()> for RetValue {
    fn from(_: ()) -> Self {
        RetValue::Null
    }
}

impl TryFrom<RetValue> for () {
    type Error = ValueDowncastError;

    fn try_from(val: RetValue) -> Result<(), Self::Error> {
        try_downcast!(val -> bool,
            |RetValue::Null| { () })
    }
}
