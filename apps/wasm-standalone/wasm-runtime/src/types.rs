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

use std::{any::TypeId, mem, slice};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum DataType {
    FP32,
    INT32,
    INT8,
}

impl DataType {
    /// Returns whether this `DataType` represents primitive type `T`.
    pub fn is_type<T: 'static>(&self) -> bool {
        let typ = TypeId::of::<T>();
        typ == TypeId::of::<i32>() || typ == TypeId::of::<i8>() || typ == TypeId::of::<f32>()
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

/// `From` conversions to `Tensor` for `ndarray::Array`.
/// Takes a reference to the `ndarray` since `Tensor` is not owned.
macro_rules! impl_tensor_from_ndarray {
    ($type:ty, $typecode:expr) => {
        impl<D: ndarray::Dimension> From<ndarray::Array<$type, D>> for Tensor {
            fn from(arr: ndarray::Array<$type, D>) -> Self {
                Tensor {
                    dtype: $typecode,
                    shape: arr.shape().iter().map(|v| *v as i64).collect(),
                    strides: Some(arr.strides().iter().map(|v| *v as usize).collect()),
                    data: unsafe {
                        slice::from_raw_parts(
                            arr.as_ptr() as *const u8,
                            arr.len() * mem::size_of::<$type>(),
                        )
                        .to_vec()
                    },
                }
            }
        }
    };
}

impl_tensor_from_ndarray!(f32, DataType::FP32);
impl_tensor_from_ndarray!(i32, DataType::INT32);
impl_tensor_from_ndarray!(i8, DataType::INT8);
