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
    os::raw::{c_int, c_void},
    slice,
};
pub use tvm_sys::ffi::DLTensor;
use tvm_sys::ffi::{
    DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDevice, DLDeviceType_kDLCPU,
};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DataType {
    FP32,
    INT32,
    INT8,
}

impl DataType {
    pub fn as_dldtype(&self) -> DLDataType {
        match self {
            DataType::INT32 => DLDataType {
                code: DLDataTypeCode_kDLInt as u8,
                bits: 32u8,
                lanes: 1u16,
            },
            DataType::INT8 => DLDataType {
                code: DLDataTypeCode_kDLInt as u8,
                bits: 8u8,
                lanes: 1u16,
            },
            DataType::FP32 => DLDataType {
                code: DLDataTypeCode_kDLFloat as u8,
                bits: 32u8,
                lanes: 1u16,
            },
        }
    }

    /// Returns whether this `DataType` represents primitive type `T`.
    pub fn is_type<T: 'static>(&self) -> bool {
        let typ = TypeId::of::<T>();
        typ == TypeId::of::<i32>() || typ == TypeId::of::<i8>() || typ == TypeId::of::<f32>()
    }
}

impl From<DLDataType> for DataType {
    fn from(dl_dtype: DLDataType) -> Self {
        if dl_dtype.code == DLDataTypeCode_kDLInt as u8 && dl_dtype.bits == 32u8 {
            DataType::INT32
        } else if dl_dtype.code == DLDataTypeCode_kDLInt as u8 && dl_dtype.bits == 8u8 {
            DataType::INT8
        } else if dl_dtype.code == DLDataTypeCode_kDLFloat as u8 && dl_dtype.bits == 32u8 {
            DataType::FP32
        } else {
            DataType::FP32
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub(crate) dtype: DataType,
    pub(crate) shape: Vec<i64>,
    pub(crate) strides: Option<Vec<usize>>,
    pub(crate) data: Vec<u8>,
}

#[allow(dead_code)]
impl Tensor {
    pub fn new(dtype: DataType, shape: Vec<i64>, strides: Vec<usize>, data: Vec<u8>) -> Self {
        Tensor {
            dtype,
            shape,
            strides: Some(strides),
            data,
        }
    }

    pub fn dtype(&self) -> DataType {
        self.dtype.clone()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    pub fn data(&self) -> Vec<u8> {
        self.data.clone()
    }

    pub fn as_dltensor(&self) -> DLTensor {
        DLTensor {
            data: self.data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType_kDLCPU,
                device_id: 0 as c_int,
            },
            ndim: self.shape.len() as c_int,
            dtype: self.dtype().as_dldtype(),
            shape: self.shape.as_ptr() as *mut i64,
            strides: self.strides.as_ref().unwrap().as_ptr() as *mut i64,
            byte_offset: 0,
            ..Default::default()
        }
    }

    /// Returns the data of this `Tensor` as a `Vec`.
    ///
    /// # Panics
    ///
    /// Panics if the `Tensor` does not contain elements of type `T`.
    pub fn to_vec<T: 'static + std::fmt::Debug + Clone>(&self) -> Vec<T> {
        assert!(self.dtype().is_type::<T>());

        unsafe {
            slice::from_raw_parts(
                self.data().as_ptr() as *const T,
                self.shape().iter().map(|v| *v as usize).product::<usize>() as usize,
            )
            .to_vec()
        }
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dtype: DataType::FP32,
            shape: Vec::new(),
            strides: None,
            data: Vec::new(),
        }
    }
}

impl From<DLTensor> for Tensor {
    fn from(dlt: DLTensor) -> Self {
        unsafe {
            let shape = slice::from_raw_parts_mut(dlt.shape, dlt.ndim as usize).to_vec();
            let size = shape.iter().map(|v| *v as usize).product::<usize>() as usize;
            let itemsize: usize = (dlt.dtype.bits >> 3).into();
            let data = slice::from_raw_parts(dlt.data as *const u8, size * itemsize).to_vec();

            Self {
                dtype: DataType::from(dlt.dtype),
                shape,
                strides: if dlt.strides.is_null() {
                    None
                } else {
                    Some(
                        slice::from_raw_parts_mut(dlt.strides as *mut usize, dlt.ndim as usize)
                            .to_vec(),
                    )
                },
                data,
            }
        }
    }
}
