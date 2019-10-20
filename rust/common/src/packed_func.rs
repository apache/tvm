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
    os::raw::c_void,
};

pub use crate::ffi::TVMValue;
use crate::{errors::ValueDowncastError, ffi::*};

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
            Null,
            Type(TVMType),
            String(CString),
            Context(TVMContext),
            Handle(*mut c_void),
            ArrayHandle(TVMArrayHandle),
            ObjectHandle(*mut c_void),
            ModuleHandle(TVMModuleHandle),
            FuncHandle(TVMFunctionHandle),
            NDArrayContainer(*mut c_void),
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
                        TVMTypeCode_kNull => Null,
                        TVMTypeCode_kTVMType => Type($value.v_type),
                        TVMTypeCode_kTVMContext => Context($value.v_ctx),
                        TVMTypeCode_kHandle => Handle($value.v_handle),
                        TVMTypeCode_kArrayHandle => ArrayHandle($value.v_handle as TVMArrayHandle),
                        TVMTypeCode_kObjectHandle => ObjectHandle($value.v_handle),
                        TVMTypeCode_kModuleHandle => ModuleHandle($value.v_handle),
                        TVMTypeCode_kFuncHandle => FuncHandle($value.v_handle),
                        TVMTypeCode_kNDArrayContainer => NDArrayContainer($value.v_handle),
                        $( $tvm_type => { $from_tvm_type } ),+
                        _ => unimplemented!("{}", type_code),
                    }
                }
            }

            pub fn to_tvm_value(&self) -> (TVMValue, TVMTypeCode) {
                use $name::*;
                match self {
                    Int(val) => (TVMValue { v_int64: *val }, DLDataTypeCode_kDLInt),
                    UInt(val) => (TVMValue { v_int64: *val as i64 }, DLDataTypeCode_kDLUInt),
                    Float(val) => (TVMValue { v_float64: *val }, DLDataTypeCode_kDLFloat),
                    Null => (TVMValue{ v_int64: 0 },TVMTypeCode_kNull),
                    Type(val) => (TVMValue { v_type: *val }, TVMTypeCode_kTVMType),
                    Context(val) => (TVMValue { v_ctx: val.clone() }, TVMTypeCode_kTVMContext),
                    String(val) => {
                        (
                            TVMValue { v_handle: val.as_ptr() as *mut c_void },
                            TVMTypeCode_kStr,
                        )
                    }
                    Handle(val) => (TVMValue { v_handle: *val }, TVMTypeCode_kHandle),
                    ArrayHandle(val) => {
                        (
                            TVMValue { v_handle: *val as *const _ as *mut c_void },
                            TVMTypeCode_kArrayHandle,
                        )
                    },
                    ObjectHandle(val) => (TVMValue { v_handle: *val }, TVMTypeCode_kObjectHandle),
                    ModuleHandle(val) =>
                        (TVMValue { v_handle: *val }, TVMTypeCode_kModuleHandle),
                    FuncHandle(val) => (
                        TVMValue { v_handle: *val },
                        TVMTypeCode_kFuncHandle
                    ),
                    NDArrayContainer(val) =>
                        (TVMValue { v_handle: *val }, TVMTypeCode_kNDArrayContainer),
                    $( $self_type($val) => { $from_self_type } ),+
                }
            }
        }
    }
}

TVMPODValue! {
    /// A borrowed TVMPODValue. Can be constructed using `into()` but the preferred way
    /// to obtain a `TVMArgValue` is automatically via `call_packed!`.
    TVMArgValue<'a> {
        Bytes(&'a TVMByteArray),
        Str(&'a CStr),
    },
    match value {
        TVMTypeCode_kBytes => { Bytes(&*(value.v_handle as *const TVMByteArray)) }
        TVMTypeCode_kStr => { Str(CStr::from_ptr(value.v_handle as *const i8)) }
    },
    match &self {
        Bytes(val) => {
            (TVMValue { v_handle: val.clone() as *const _ as *mut c_void }, TVMTypeCode_kBytes)
        }
        Str(val) => { (TVMValue { v_handle: val.as_ptr() as *mut c_void }, TVMTypeCode_kStr) }
    }
}

TVMPODValue! {
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
    TVMRetValue {
        Bytes(TVMByteArray),
        Str(&'static CStr),
    },
    match value {
        TVMTypeCode_kBytes => { Bytes(*(value.v_handle as *const TVMByteArray)) }
        TVMTypeCode_kStr => { Str(CStr::from_ptr(value.v_handle as *mut i8)) }
    },
    match &self {
        Bytes(val) =>
            { (TVMValue { v_handle: val as *const _ as *mut c_void }, TVMTypeCode_kBytes ) }
        Str(val) =>
            { (TVMValue { v_str: val.as_ptr() }, TVMTypeCode_kStr ) }
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

/// Creates a conversion to a `TVMArgValue` for a primitive type and DLDataTypeCode.
macro_rules! impl_pod_value {
    ($variant:ident, $inner_ty:ty, [ $( $type:ty ),+ ] ) => {
        $(
            impl<'a> From<$type> for TVMArgValue<'a> {
                fn from(val: $type) -> Self {
                    Self::$variant(val as $inner_ty)
                }
            }

            impl<'a, 'v> From<&'a $type> for TVMArgValue<'v> {
                fn from(val: &'a $type) -> Self {
                    Self::$variant(*val as $inner_ty)
                }
            }

            impl<'a> TryFrom<TVMArgValue<'a>> for $type {
                type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: TVMArgValue<'a>) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |TVMArgValue::$variant(val)| { val as $type })
                }
            }

            impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for $type {
                type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: &'a TVMArgValue<'v>) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |TVMArgValue::$variant(val)| { *val as $type })
                }
            }

            impl From<$type> for TVMRetValue {
                fn from(val: $type) -> Self {
                    Self::$variant(val as $inner_ty)
                }
            }

            impl TryFrom<TVMRetValue> for $type {
              type Error = $crate::errors::ValueDowncastError;
                fn try_from(val: TVMRetValue) -> Result<Self, Self::Error> {
                    try_downcast!(val -> $type, |TVMRetValue::$variant(val)| { val as $type })
                }
            }
        )+
    };
}

impl_pod_value!(Int, i64, [i8, i16, i32, i64, isize]);
impl_pod_value!(UInt, i64, [u8, u16, u32, u64, usize]);
impl_pod_value!(Float, f64, [f32, f64]);
impl_pod_value!(Type, TVMType, [TVMType]);
impl_pod_value!(Context, TVMContext, [TVMContext]);

impl<'a> From<&'a str> for TVMArgValue<'a> {
    fn from(s: &'a str) -> Self {
        Self::String(CString::new(s).unwrap())
    }
}

impl<'a> From<String> for TVMArgValue<'a> {
    fn from(s: String) -> Self {
        Self::String(CString::new(s).unwrap())
    }
}

impl<'a> From<&'a CStr> for TVMArgValue<'a> {
    fn from(s: &'a CStr) -> Self {
        Self::Str(s)
    }
}

impl<'a> From<&'a TVMByteArray> for TVMArgValue<'a> {
    fn from(s: &'a TVMByteArray) -> Self {
        Self::Bytes(s)
    }
}

impl<'a> TryFrom<TVMArgValue<'a>> for &'a str {
    type Error = ValueDowncastError;
    fn try_from(val: TVMArgValue<'a>) -> Result<Self, Self::Error> {
        try_downcast!(val -> &str, |TVMArgValue::Str(s)| { s.to_str().unwrap() })
    }
}

impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for &'v str {
    type Error = ValueDowncastError;
    fn try_from(val: &'a TVMArgValue<'v>) -> Result<Self, Self::Error> {
        try_downcast!(val -> &str, |TVMArgValue::Str(s)| { s.to_str().unwrap() })
    }
}

/// Converts an unspecialized handle to a TVMArgValue.
impl<T> From<*const T> for TVMArgValue<'static> {
    fn from(ptr: *const T) -> Self {
        Self::Handle(ptr as *mut c_void)
    }
}

/// Converts an unspecialized mutable handle to a TVMArgValue.
impl<T> From<*mut T> for TVMArgValue<'static> {
    fn from(ptr: *mut T) -> Self {
        Self::Handle(ptr as *mut c_void)
    }
}

impl<'a> From<&'a mut DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a mut DLTensor) -> Self {
        Self::ArrayHandle(arr as *mut DLTensor)
    }
}

impl<'a> From<&'a DLTensor> for TVMArgValue<'a> {
    fn from(arr: &'a DLTensor) -> Self {
        Self::ArrayHandle(arr as *const _ as *mut DLTensor)
    }
}

impl TryFrom<TVMRetValue> for String {
    type Error = ValueDowncastError;
    fn try_from(val: TVMRetValue) -> Result<String, Self::Error> {
        try_downcast!(
            val -> String,
            |TVMRetValue::String(s)| { s.into_string().unwrap() },
            |TVMRetValue::Str(s)| { s.to_str().unwrap().to_string() }
        )
    }
}

impl From<String> for TVMRetValue {
    fn from(s: String) -> Self {
        Self::String(std::ffi::CString::new(s).unwrap())
    }
}

impl From<TVMByteArray> for TVMRetValue {
    fn from(arr: TVMByteArray) -> Self {
        Self::Bytes(arr)
    }
}

impl TryFrom<TVMRetValue> for TVMByteArray {
    type Error = ValueDowncastError;
    fn try_from(val: TVMRetValue) -> Result<Self, Self::Error> {
        try_downcast!(val -> TVMByteArray, |TVMRetValue::Bytes(val)| { val })
    }
}

impl Default for TVMRetValue {
    fn default() -> Self {
        Self::Int(0)
    }
}
