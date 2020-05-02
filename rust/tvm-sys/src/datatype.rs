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
    any::TypeId,
    mem,
    os::raw::{c_int, c_void},
};

use std::str::FromStr;
use thiserror::Error;

use crate::ffi::{
    DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
    DLDeviceType_kDLCPU, DLTensor,
};


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

impl DataType {
    pub fn new(code: u8, bits: u8, lanes: u16) -> DataType {
        DataType { code, bits, lanes }
    }

    /// Returns the number of bytes occupied by an element of this `DataType`.
    pub fn itemsize(&self) -> usize {
        (self.bits as usize * self.lanes as usize) >> 3
    }

    /// Returns whether this `DataType` represents primitive type `T`.
    pub fn is_type<T: 'static>(&self) -> bool {
        if self.lanes != 1 {
            return false;
        }
        let typ = TypeId::of::<T>();
        (typ == TypeId::of::<i32>() && self.code == 0 && self.bits == 32)
            || (typ == TypeId::of::<i64>() && self.code == 0 && self.bits == 64)
            || (typ == TypeId::of::<u32>() && self.code == 1 && self.bits == 32)
            || (typ == TypeId::of::<u64>() && self.code == 1 && self.bits == 64)
            || (typ == TypeId::of::<f32>() && self.code == 2 && self.bits == 32)
            || (typ == TypeId::of::<f64>() && self.code == 2 && self.bits == 64)
    }

    pub fn code(&self) -> usize {
        self.code as usize
    }

    pub fn bits(&self) -> usize {
        self.bits as usize
    }

    pub fn lanes(&self) -> usize {
        self.lanes as usize
    }
}

impl<'a> From<&'a DataType> for DLDataType {
    fn from(dtype: &'a DataType) -> Self {
        Self {
            code: dtype.code as u8,
            bits: dtype.bits as u8,
            lanes: dtype.lanes as u16,
        }
    }
}

impl From<DLDataType> for DataType {
    fn from(dtype: DLDataType) -> Self {
        Self {
            code: dtype.code,
            bits: dtype.bits,
            lanes: dtype.lanes,
        }
    }
}

#[derive(Debug, Error)]
pub enum ParseTvmTypeError {
    #[error("invalid number: {0}")]
    InvalidNumber(std::num::ParseIntError),
    #[error("unknown type: {0}")]
    UnknownType(String),
}

/// Implements TVMType conversion from `&str` of general format `{dtype}{bits}x{lanes}`
/// such as "int32", "float32" or with lane "float32x1".
impl FromStr for DataType {
    type Err = ParseTvmTypeError;
    fn from_str(type_str: &str) -> Result<Self, Self::Err> {
        if type_str == "bool" {
            return Ok(DataType::new(1, 1, 1));
        }

        let mut type_lanes = type_str.split('x');
        let typ = type_lanes.next().expect("Missing dtype");
        let lanes = type_lanes
            .next()
            .map(|l| <u16>::from_str_radix(l, 10))
            .unwrap_or(Ok(1))
            .map_err(ParseTvmTypeError::InvalidNumber)?;
        let (type_name, bits) = match typ.find(char::is_numeric) {
            Some(idx) => {
                let (name, bits_str) = typ.split_at(idx);
                (
                    name,
                    u8::from_str_radix(bits_str, 10).map_err(ParseTvmTypeError::InvalidNumber)?,
                )
            }
            None => (typ, 32),
        };

        let type_code = match type_name {
            "int" => 0,
            "uint" => 1,
            "float" => 2,
            "handle" => 3,
            _ => return Err(ParseTvmTypeError::UnknownType(type_name.to_string())),
        };

        Ok(DataType::new(type_code, bits, lanes))
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.bits == 1 && self.lanes == 1 {
            return write!(f, "bool");
        }
        let mut type_str = match self.code {
            0 => "int",
            1 => "uint",
            2 => "float",
            4 => "handle",
            _ => "unknown",
        }
        .to_string();

        type_str += &self.bits.to_string();
        if self.lanes > 1 {
            type_str += &format!("x{}", self.lanes);
        }
        f.write_str(&type_str)
    }
}

/// `From` conversions to `DLTensor` for `ndarray::Array`.
/// Takes a reference to the `ndarray` since `DLTensor` is not owned.
macro_rules! impl_dltensor_from_ndarray {
    ($type:ty, $typecode:expr) => {
        impl<'a, D: ndarray::Dimension> From<&'a mut ndarray::Array<$type, D>> for DLTensor {
            fn from(arr: &'a mut ndarray::Array<$type, D>) -> Self {
                DLTensor {
                    data: arr.as_mut_ptr() as *mut c_void,
                    ctx: DLContext {
                        device_type: DLDeviceType_kDLCPU,
                        device_id: 0,
                    },
                    ndim: arr.ndim() as c_int,
                    dtype: DLDataType {
                        code: $typecode as u8,
                        bits: 8 * mem::size_of::<$type>() as u8,
                        lanes: 1,
                    },
                    shape: arr.shape().as_ptr() as *const i64 as *mut i64,
                    strides: arr.strides().as_ptr() as *const isize as *mut i64,
                    byte_offset: 0,
                }
            }
        }
    };
}

impl_dltensor_from_ndarray!(f32, DLDataTypeCode_kDLFloat);
impl_dltensor_from_ndarray!(f64, DLDataTypeCode_kDLFloat);
impl_dltensor_from_ndarray!(i32, DLDataTypeCode_kDLInt);
impl_dltensor_from_ndarray!(i64, DLDataTypeCode_kDLInt);
impl_dltensor_from_ndarray!(u32, DLDataTypeCode_kDLUInt);
impl_dltensor_from_ndarray!(u64, DLDataTypeCode_kDLUInt);
