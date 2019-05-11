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

use std::{os::raw::c_char, str::FromStr};

use failure::Error;

use crate::ffi::*;

impl TVMType {
    fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
        Self {
            code: type_code,
            bits,
            lanes,
        }
    }
}

/// Implements TVMType conversion from `&str` of general format `{dtype}{bits}x{lanes}`
/// such as "int32", "float32" or with lane "float32x1".
impl FromStr for TVMType {
    type Err = Error;
    fn from_str(type_str: &str) -> Result<Self, Self::Err> {
        if type_str == "bool" {
            return Ok(TVMType::new(1, 1, 1));
        }

        let mut type_lanes = type_str.split("x");
        let typ = type_lanes.next().expect("Missing dtype");
        let lanes = type_lanes
            .next()
            .map(|l| <u16>::from_str_radix(l, 10))
            .unwrap_or(Ok(1))?;
        let (type_name, bits) = match typ.find(char::is_numeric) {
            Some(idx) => {
                let (name, bits_str) = typ.split_at(idx);
                (name, u8::from_str_radix(bits_str, 10)?)
            }
            None => (typ, 32),
        };

        let type_code = match type_name {
            "int" => 0,
            "uint" => 1,
            "float" => 2,
            "handle" => 3,
            _ => return Err(format_err!("Unknown type {}", type_name)),
        };

        Ok(TVMType::new(type_code, bits, lanes))
    }
}

impl std::fmt::Display for TVMType {
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

macro_rules! impl_pod_tvm_value {
    ($field:ident, $field_ty:ty, $( $ty:ty ),+) => {
        $(
            impl From<$ty> for TVMValue {
                fn from(val: $ty) -> Self {
                    TVMValue { $field: val as $field_ty }
                }
            }

            impl From<TVMValue> for $ty {
                fn from(val: TVMValue) -> Self {
                    unsafe { val.$field as $ty }
                }
            }
        )+
    };
    ($field:ident, $ty:ty) => {
        impl_pod_tvm_value!($field, $ty, $ty);
    }
}

impl_pod_tvm_value!(v_int64, i64, i8, u8, i16, u16, i32, u32, i64, u64, isize, usize);
impl_pod_tvm_value!(v_float64, f64, f32, f64);
impl_pod_tvm_value!(v_type, TVMType);
impl_pod_tvm_value!(v_ctx, TVMContext);

macro_rules! impl_tvm_context {
    ( $( $dev_type:ident : [ $( $dev_name:ident ),+ ] ),+ ) => {
        /// Creates a TVMContext from a string (e.g., "cpu", "gpu", "ext_dev")
        impl FromStr for TVMContext {
            type Err = Error;
            fn from_str(type_str: &str) -> Result<Self, Self::Err> {
                Ok(Self {
                    device_type: match type_str {
                         $( $(  stringify!($dev_name)  )|+ => $dev_type ),+,
                        _ => return Err(format_err!("device {} not supported", type_str).into()),
                    },
                    device_id: 0,
                })
            }
        }

        impl TVMContext {
            $(
                $(
                    pub fn $dev_name(device_id: usize) -> Self {
                        Self {
                            device_type: $dev_type,
                            device_id: device_id as i32,
                        }
                    }
                )+
            )+
        }
    };
}

impl_tvm_context!(
    DLDeviceType_kDLCPU: [cpu, llvm, stackvm],
    DLDeviceType_kDLGPU: [gpu, cuda, nvptx],
    DLDeviceType_kDLOpenCL: [cl],
    DLDeviceType_kDLMetal: [metal],
    DLDeviceType_kDLVPI: [vpi],
    DLDeviceType_kDLROCM: [rocm],
    DLDeviceType_kDLExtDev: [ext_dev]
);

/// A struct holding TVM byte-array.
///
/// ## Example
///
/// ```
/// let v = b"hello";
/// let barr = TVMByteArray::from(&v);
/// assert_eq!(barr.len(), v.len());
/// assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
/// ```
impl TVMByteArray {
    /// Gets the underlying byte-array
    pub fn data(&self) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.data as *const u8, self.size) }
    }

    /// Gets the length of the underlying byte-array
    pub fn len(&self) -> usize {
        self.size
    }

    /// Converts the underlying byte-array to `Vec<u8>`
    pub fn to_vec(&self) -> Vec<u8> {
        self.data().to_vec()
    }
}

// Needs AsRef for Vec
impl<T: AsRef<[u8]>> From<T> for TVMByteArray {
    fn from(arg: T) -> Self {
        let arg = arg.as_ref();
        TVMByteArray {
            data: arg.as_ptr() as *const c_char,
            size: arg.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        let v = vec![1u8, 2, 3];
        let barr = TVMByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.to_vec(), vec![1u8, 2, 3]);
        let v = b"hello";
        let barr = TVMByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
    }
}
