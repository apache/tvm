//! This module implements the required conversions from Rust types to TVM types.
//!
//! In TVM frontend only conversions from Rust's 32-bits (POD) numeric types (i32, u32, f32)
//! and 64-bits pointers are supported.

use std::{
    fmt::{self, Display, Formatter},
    ops::{Deref, DerefMut},
};

use crate::ts;

use crate::{Function, Module, NDArray, TVMByteArray, TVMContext, TVMDeviceType, TVMTypeCode};

macro_rules! impl_prim_type {
    ($type:ty, $variant:ident) => {
        impl From<$type> for TVMTypeCode {
            fn from(_arg: $type) -> Self {
                TVMTypeCode::$variant
            }
        }

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

impl_prim_type!(TVMDeviceType, kDLInt);
impl_prim_type!(TVMContext, kTVMContext);
impl_prim_type!(TVMType, kTVMType);
impl_prim_type!(Function, kFuncHandle);
impl_prim_type!(Module, kModuleHandle);
impl_prim_type!(NDArray, kArrayHandle);
impl_prim_type!(TVMByteArray, kBytes);

/// See the [module-level documentation](../ty/index.html) for more details.
///
/// Wrapper around underlying TVMType
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TVMType {
    // inner fields are (code: u8, bits: u8, lanes: u16)
    pub inner: ts::TVMType,
}

impl TVMType {
    pub(crate) fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
        TVMType {
            inner: ts::TVMType {
                code: type_code,
                bits: bits,
                lanes: lanes,
            },
        }
    }
}

/// Implements TVMType conversion from `&str` of general format `{dtype}{bits}x{lanes}`
/// such as "int32", "float32" or with lane "float32x1".
impl<'a> From<&'a str> for TVMType {
    fn from(type_str: &'a str) -> Self {
        if type_str == "bool" {
            return TVMType::new(1, 1, 1);
        }

        let mut type_lanes = type_str.split("x");
        let typ = type_lanes.next().expect("Missing dtype");
        let lanes = type_lanes
            .next()
            .map(|l| u16::from_str_radix(l, 10).expect(&format!("Bad dtype lanes: {}", l)))
            .unwrap_or(1);
        let (type_name, bits) = match typ.find(char::is_numeric) {
            Some(idx) => {
                let (name, bits_str) = typ.split_at(idx);
                (
                    name,
                    u8::from_str_radix(bits_str, 10)
                        .expect(&format!("Bad dtype bits: {}", bits_str)),
                )
            }
            None => (typ, 32),
        };

        let type_code = match type_name {
            "int" => 0,
            "uint" => 1,
            "float" => 2,
            "handle" => 3,
            _ => unimplemented!(),
        };

        TVMType::new(type_code, bits, lanes)
    }
}

impl Display for TVMType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let ts::TVMType { code, bits, lanes } = self.inner;
        if bits == 1 && lanes == 1 {
            return write!(f, "bool");
        }
        let mut tcode_str = match code {
            0 => "int",
            1 => "uint",
            2 => "float",
            4 => "handle",
            _ => "Unknown",
        }
        .to_string();

        tcode_str += &bits.to_string();
        if lanes > 1 {
            tcode_str += &format!("x{}", lanes.to_string());
        }
        f.write_str(&tcode_str)
    }
}

impl From<TVMType> for ts::DLDataType {
    fn from(dtype: TVMType) -> Self {
        dtype.inner
    }
}

impl From<ts::DLDataType> for TVMType {
    fn from(dtype: ts::DLDataType) -> Self {
        Self::new(dtype.code, dtype.bits, dtype.lanes)
    }
}

impl Deref for TVMType {
    type Target = ts::TVMType;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TVMType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
