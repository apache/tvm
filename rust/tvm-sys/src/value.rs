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

use std::str::FromStr;

use crate::ffi::*;

use thiserror::Error;

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
impl_pod_tvm_value!(v_type, DLDataType);
impl_pod_tvm_value!(v_device, DLDevice);

#[derive(Debug, Error)]
#[error("unsupported device: {0}")]
pub struct UnsupportedDeviceError(String);

macro_rules! impl_tvm_device {
    ( $( $dev_type:ident : [ $( $dev_name:ident ),+ ] ),+ ) => {
        /// Creates a DLDevice from a string (e.g., "cpu", "cuda", "ext_dev")
        impl FromStr for DLDevice {
            type Err = UnsupportedDeviceError;
            fn from_str(type_str: &str) -> Result<Self, Self::Err> {
                Ok(Self {
                    device_type: match type_str {
                         $( $(  stringify!($dev_name)  )|+ => $dev_type ),+,
                        _ => return Err(UnsupportedDeviceError(type_str.to_string())),
                    },
                    device_id: 0,
                })
            }
        }

        impl DLDevice {
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

impl_tvm_device!(
    DLDeviceType_kDLCPU: [cpu, llvm, stackvm],
    DLDeviceType_kDLCUDA: [cuda, nvptx],
    DLDeviceType_kDLOpenCL: [cl],
    DLDeviceType_kDLMetal: [metal],
    DLDeviceType_kDLVPI: [vpi],
    DLDeviceType_kDLROCM: [rocm],
    DLDeviceType_kDLExtDev: [ext_dev]
);
