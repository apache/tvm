//! This module containes `TVMTypeCode` and `TVMType` with some conversion methods.
//!
//! # Example
//!
//! ```
//! let dtype = TVMType::from("float");
//! println!("dtype is: {}", dtype);
//! ```

use std::{
    ffi::{CStr, CString},
    fmt::{self, Display, Formatter},
};

/// TVM type codes.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TVMTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kHandle = 3,
    kNull = 4,
    kTVMType = 5,
    kTVMContext = 6,
    kArrayHandle = 7,
    kNodeHandle = 8,
    kModuleHandle = 9,
    kFuncHandle = 10,
    kStr = 11,
    kBytes = 12,
    kNDArrayContainer = 13,
}

impl Default for TVMTypeCode {
    fn default() -> Self {
        TVMTypeCode::kDLInt
    }
}

impl From<TVMTypeCode> for i64 {
    fn from(arg: TVMTypeCode) -> i64 {
        match arg {
            TVMTypeCode::kDLInt => 0,
            TVMTypeCode::kDLUInt => 1,
            TVMTypeCode::kDLFloat => 2,
            TVMTypeCode::kHandle => 3,
            TVMTypeCode::kNull => 4,
            TVMTypeCode::kTVMType => 5,
            TVMTypeCode::kTVMContext => 6,
            TVMTypeCode::kArrayHandle => 7,
            TVMTypeCode::kNodeHandle => 8,
            TVMTypeCode::kModuleHandle => 9,
            TVMTypeCode::kFuncHandle => 10,
            TVMTypeCode::kStr => 11,
            TVMTypeCode::kBytes => 12,
            TVMTypeCode::kNDArrayContainer => 13,
        }
    }
}

impl Into<TVMTypeCode> for i64 {
    fn into(self) -> TVMTypeCode {
        match self {
            0 => TVMTypeCode::kDLInt,
            1 => TVMTypeCode::kDLUInt,
            2 => TVMTypeCode::kDLFloat,
            3 => TVMTypeCode::kHandle,
            4 => TVMTypeCode::kNull,
            5 => TVMTypeCode::kTVMType,
            6 => TVMTypeCode::kTVMContext,
            7 => TVMTypeCode::kArrayHandle,
            8 => TVMTypeCode::kNodeHandle,
            9 => TVMTypeCode::kModuleHandle,
            10 => TVMTypeCode::kFuncHandle,
            11 => TVMTypeCode::kStr,
            12 => TVMTypeCode::kBytes,
            13 => TVMTypeCode::kNDArrayContainer,
            _ => unreachable!(),
        }
    }
}

impl Display for TVMTypeCode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TVMTypeCode::kDLInt => "int",
                TVMTypeCode::kDLUInt => "uint",
                TVMTypeCode::kDLFloat => "float",
                TVMTypeCode::kHandle => "handle",
                TVMTypeCode::kNull => "null",
                TVMTypeCode::kTVMType => "TVM type",
                TVMTypeCode::kTVMContext => "TVM context",
                TVMTypeCode::kArrayHandle => "Array handle",
                TVMTypeCode::kNodeHandle => "Node handle",
                TVMTypeCode::kModuleHandle => "Module handle",
                TVMTypeCode::kFuncHandle => "Function handle",
                TVMTypeCode::kStr => "string",
                TVMTypeCode::kBytes => "bytes",
                TVMTypeCode::kNDArrayContainer => "ndarray container",
            }
        )
    }
}

macro_rules! impl_prim_type {
    ($type:ty, $variant:ident) => {
        impl<'a> From<&'a $type> for TVMTypeCode {
            fn from(_arg: &$type) -> Self {
                TVMTypeCode::$variant
            }
        }

        impl<'a> From<&'a mut $type> for TVMTypeCode {
            fn from(_arg: &mut $type) -> Self {
                TVMTypeCode::$variant
            }
        }
    };
}

impl_prim_type!(usize, kDLInt);
impl_prim_type!(i64, kDLInt);
impl_prim_type!(i32, kDLInt);
impl_prim_type!(i16, kDLInt);
impl_prim_type!(i8, kDLInt);

impl_prim_type!(u64, kDLUInt);
impl_prim_type!(u32, kDLUInt);
impl_prim_type!(u16, kDLUInt);
impl_prim_type!(u8, kDLUInt);

impl_prim_type!(f64, kDLFloat);
impl_prim_type!(f32, kDLFloat);

impl_prim_type!(str, kStr);
impl_prim_type!(CStr, kStr);
impl_prim_type!(String, kStr);
impl_prim_type!(CString, kStr);

impl_prim_type!([u8], kBytes);
