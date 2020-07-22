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

use crate::ffi::{
    DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
    DLDeviceType_kDLCPU, DLTensor,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataType {
    pub code: usize,
    pub bits: usize,
    pub lanes: usize,
}

impl DataType {
    /// Returns the number of bytes occupied by an element of this `DataType`.
    pub fn itemsize(&self) -> usize {
        (self.bits * self.lanes) >> 3
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
        self.code
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn lanes(&self) -> usize {
        self.lanes
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
            code: dtype.code as usize,
            bits: dtype.bits as usize,
            lanes: dtype.lanes as usize,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TVMContext {
    pub device_type: usize,
    pub device_id: usize,
}

impl<'a> From<&'a TVMContext> for DLContext {
    fn from(ctx: &'a TVMContext) -> Self {
        Self {
            device_type: ctx.device_type as _,
            device_id: ctx.device_id as i32,
        }
    }
}

impl Default for TVMContext {
    fn default() -> Self {
        Self {
            device_type: DLDeviceType_kDLCPU as usize,
            device_id: 0,
        }
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
