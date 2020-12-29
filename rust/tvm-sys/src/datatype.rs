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

use std::any::TypeId;
use std::convert::TryFrom;
use std::str::FromStr;

use crate::ffi::DLDataType;
use crate::packed_func::RetValue;

use thiserror::Error;

const DL_INT_CODE: u8 = 0;
const DL_UINT_CODE: u8 = 1;
const DL_FLOAT_CODE: u8 = 2;
const DL_HANDLE: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct DataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

impl DataType {
    pub const fn new(code: u8, bits: u8, lanes: u16) -> DataType {
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
        (typ == TypeId::of::<i32>() && self.code == DL_INT_CODE && self.bits == 32)
            || (typ == TypeId::of::<i64>() && self.code == DL_INT_CODE && self.bits == 64)
            || (typ == TypeId::of::<u32>() && self.code == DL_UINT_CODE && self.bits == 32)
            || (typ == TypeId::of::<u64>() && self.code == DL_UINT_CODE && self.bits == 64)
            || (typ == TypeId::of::<f32>() && self.code == DL_FLOAT_CODE && self.bits == 32)
            || (typ == TypeId::of::<f64>() && self.code == DL_FLOAT_CODE && self.bits == 64)
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

    pub const fn int(bits: u8, lanes: u16) -> DataType {
        DataType::new(DL_INT_CODE, bits, lanes)
    }

    pub const fn float(bits: u8, lanes: u16) -> DataType {
        DataType::new(DL_FLOAT_CODE, bits, lanes)
    }

    pub const fn float32() -> DataType {
        Self::float(32, 1)
    }

    pub const fn uint(bits: u8, lanes: u16) -> DataType {
        DataType::new(DL_UINT_CODE, bits, lanes)
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

impl From<DataType> for DLDataType {
    fn from(dtype: DataType) -> Self {
        Self {
            code: dtype.code,
            bits: dtype.bits,
            lanes: dtype.lanes,
        }
    }
}

#[derive(Debug, Error)]
pub enum ParseDataTypeError {
    #[error("invalid number: {0}")]
    InvalidNumber(std::num::ParseIntError),
    #[error("missing data type specifier (e.g., int32, float64)")]
    MissingDataType,
    #[error("unknown type: {0}")]
    UnknownType(String),
}

/// Implements TVMType conversion from `&str` of general format `{dtype}{bits}x{lanes}`
/// such as "int32", "float32" or with lane "float32x1".
impl FromStr for DataType {
    type Err = ParseDataTypeError;

    fn from_str(type_str: &str) -> Result<Self, Self::Err> {
        use ParseDataTypeError::*;

        if type_str == "bool" {
            return Ok(DataType::new(1, 1, 1));
        }

        let mut type_lanes = type_str.split('x');
        let typ = type_lanes.next().ok_or(MissingDataType)?;
        let lanes = type_lanes
            .next()
            .map(|l| <u16>::from_str_radix(l, 10))
            .unwrap_or(Ok(1))
            .map_err(InvalidNumber)?;
        let (type_name, bits) = match typ.find(char::is_numeric) {
            Some(idx) => {
                let (name, bits_str) = typ.split_at(idx);
                (
                    name,
                    u8::from_str_radix(bits_str, 10).map_err(InvalidNumber)?,
                )
            }
            None => (typ, 32),
        };

        let type_code = match type_name {
            "int" => DL_INT_CODE,
            "uint" => DL_UINT_CODE,
            "float" => DL_FLOAT_CODE,
            "handle" => DL_HANDLE,
            _ => return Err(UnknownType(type_name.to_string())),
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
            DL_INT_CODE => "int",
            DL_UINT_CODE => "uint",
            DL_FLOAT_CODE => "float",
            DL_HANDLE => "handle",
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

impl From<DataType> for RetValue {
    fn from(dt: DataType) -> RetValue {
        RetValue::DataType((&dt).into())
    }
}

impl TryFrom<RetValue> for DataType {
    type Error = anyhow::Error;
    fn try_from(ret_value: RetValue) -> anyhow::Result<DataType> {
        match ret_value {
            RetValue::DataType(dt) => Ok(dt.into()),
            // TODO(@jroesch): improve
            _ => Err(anyhow::anyhow!("unable to convert datatype from ...")),
        }
    }
}
