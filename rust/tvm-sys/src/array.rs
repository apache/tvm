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
    mem,
    os::raw::{c_int, c_void},
};

use crate::ffi::{
    DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt, DLDevice,
    DLDeviceType_kDLCPU, DLTensor,
};

/// `From` conversions to `DLTensor` for `ndarray::Array`.
/// Takes a reference to the `ndarray` since `DLTensor` is not owned.
macro_rules! impl_dltensor_from_ndarray {
    ($type:ty, $typecode:expr) => {
        impl<'a, D: ndarray::Dimension> From<&'a mut ndarray::Array<$type, D>> for DLTensor {
            fn from(arr: &'a mut ndarray::Array<$type, D>) -> Self {
                DLTensor {
                    data: arr.as_mut_ptr() as *mut c_void,
                    device: DLDevice {
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
                    strides: arr.strides().as_ptr() as *const i64 as *mut i64,
                    byte_offset: 0,
                    ..Default::default()
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
