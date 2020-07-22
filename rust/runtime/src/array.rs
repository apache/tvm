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

use std::{convert::TryFrom, mem, os::raw::c_void, ptr, slice};

use failure::Error;
use ndarray;
use tvm_common::{
    array::{DataType, TVMContext},
    ffi::{
        DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt,
        DLDataTypeCode_kDLUInt, DLTensor,
    },
};

use crate::allocator::Allocation;

/// A `Storage` is a container which holds `Tensor` data.
#[derive(PartialEq)]
pub enum Storage<'a> {
    /// A `Storage` which owns its contained bytes.
    Owned(Allocation),

    /// A view of an existing `Storage`.
    View(&'a mut [u8], usize), // ptr, align
}

impl<'a> Storage<'a> {
    pub fn new(size: usize, align: Option<usize>) -> Result<Storage<'static>, Error> {
        Ok(Storage::Owned(Allocation::new(size, align)?))
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        match self {
            Storage::Owned(alloc) => alloc.as_mut_ptr(),
            Storage::View(slice, _) => slice.as_ptr() as *mut u8,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Storage::Owned(alloc) => alloc.size(),
            Storage::View(slice, _) => slice.len(),
        }
    }

    pub fn align(&self) -> usize {
        match self {
            Storage::Owned(alloc) => alloc.align(),
            Storage::View(_, align) => *align,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.as_mut_ptr() as *const _
    }

    /// Returns a `Storage::View` which points to an owned `Storage::Owned`.
    pub fn view(&self) -> Storage<'a> {
        match self {
            Storage::Owned(alloc) => Storage::View(
                unsafe { slice::from_raw_parts_mut(alloc.as_mut_ptr(), self.size()) },
                self.align(),
            ),
            Storage::View(slice, _) => Storage::View(
                unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), slice.len()) },
                self.align(),
            ),
        }
    }

    pub fn is_owned(&self) -> bool {
        match self {
            Storage::Owned(_) => true,
            _ => false,
        }
    }

    /// Returns an owned version of this storage via cloning.
    pub fn to_owned(&self) -> Storage<'static> {
        let s = Storage::new(self.size(), Some(self.align())).unwrap();
        unsafe {
            s.as_mut_ptr()
                .copy_from_nonoverlapping(self.as_ptr(), self.size());
        }
        s
    }

    /// Returns a view of the stored data.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Storage::Owned(alloc) => alloc.as_slice(),
            Storage::View(slice, _) => &*slice,
        }
    }

    /// Returns a mutable view of the stored data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Storage::Owned(alloc) => alloc.as_mut_slice(),
            Storage::View(slice, _) => slice,
        }
    }
}

impl<'d, 's, T> From<&'d [T]> for Storage<'s> {
    fn from(data: &'d [T]) -> Self {
        let data = unsafe {
            slice::from_raw_parts_mut(
                data.as_ptr() as *const u8 as *mut u8,
                data.len() * mem::size_of::<T>() as usize,
            )
        };
        Storage::View(data, mem::align_of::<T>())
    }
}

/// A n-dimensional array type which can be converted to/from `tvm::DLTensor` and `ndarray::Array`.
/// `Tensor` is primarily a holder of data which can be operated on via TVM (via `DLTensor`) or
/// converted to `ndarray::Array` for non-TVM processing.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// use std::convert::TryInto;
/// use tvm_runtime::{call_packed, DLTensor, TVMArgValue, TVMRetValue, Tensor};
///
/// let mut a_nd: ndarray::Array1<f32> = ndarray::Array::from_vec(vec![1f32, 2., 3., 4.]);
/// let mut a: Tensor = a_nd.into();
/// let mut a_dl: DLTensor = (&mut a).into();
///
/// let tvm_fn = |args: &[TVMArgValue]| -> Result<TVMRetValue, ()> { Ok(TVMRetValue::default()) };
/// call_packed!(tvm_fn, &mut a_dl);
///
/// // Array -> Tensor is mostly useful when post-processing TVM graph outputs.
/// let mut a_nd: ndarray::ArrayD<f32> = a.try_into().unwrap();
/// ```
#[derive(PartialEq)]
pub struct Tensor<'a> {
    /// The bytes which contain the data this `Tensor` represents.
    pub(crate) data: Storage<'a>,
    pub(crate) ctx: TVMContext,
    pub(crate) dtype: DataType,
    pub(crate) shape: Vec<i64>,
    // ^ not usize because `typedef int64_t tvm_index_t` in c_runtime_api.h
    /// The `Tensor` strides. Can be `None` if the `Tensor` is contiguous.
    pub(crate) strides: Option<Vec<usize>>,
    pub(crate) byte_offset: isize,
    /// The number of elements in the `Tensor`.
    pub(crate) size: usize,
}

unsafe impl<'a> Send for Tensor<'a> {}

impl<'a> Tensor<'a> {
    pub fn shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    pub fn data(&self) -> &Storage {
        &self.data
    }

    pub fn data_mut(&mut self) -> &'a mut Storage {
        &mut self.data
    }

    /// Returns the data of this `Tensor` as a `Vec`.
    ///
    /// # Panics
    ///
    /// Panics if the `Tensor` is not contiguous or does not contain elements of type `T`.
    pub fn to_vec<T: 'static + std::fmt::Debug + Clone>(&self) -> Vec<T> {
        assert!(self.is_contiguous());
        assert!(self.dtype.is_type::<T>());
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.size).to_vec() }
    }

    /// Returns `true` iff this `Tensor` is represented by a contiguous region of memory.
    pub fn is_contiguous(&self) -> bool {
        match self.strides {
            None => true,
            Some(ref strides) => {
                // check that stride for each dimension is the
                // product of all trailing dimensons' shapes
                self.shape
                    .iter()
                    .zip(strides)
                    .rfold(
                        (true, 1),
                        |(is_contig, expected_stride), (shape, stride)| {
                            (
                                is_contig && *stride == expected_stride,
                                expected_stride * (*shape as usize),
                            )
                        },
                    )
                    .0
            }
        }
    }

    /// Returns a clone of this `Tensor`.
    ///
    /// # Panics
    ///
    /// Panics if the `Tensor` is not contiguous or does not contain elements of type `T`.
    pub fn copy(&mut self, other: &Tensor) {
        assert!(
            self.dtype == other.dtype && self.size == other.size,
            "Tensor shape/dtype mismatch."
        );
        assert!(
      self.is_contiguous() && other.is_contiguous(),
      "copy currently requires contiguous tensors\n`self.strides = {:?}` `other.strides = {:?}`",
      self.strides,
      other.strides
    );
        unsafe {
            self.data
                .as_mut_ptr()
                .offset(self.byte_offset as isize)
                .copy_from_nonoverlapping(
                    other.data.as_mut_ptr().offset(other.byte_offset),
                    other.size * other.dtype.itemsize(),
                );
        }
    }

    /// Returns an owned version of this `Tensor` via cloning.
    pub fn to_owned(&self) -> Tensor<'static> {
        let t = Tensor {
            data: self.data.to_owned(),
            ctx: self.ctx,
            dtype: self.dtype,
            size: self.size,
            shape: self.shape.clone(),
            strides: None,
            byte_offset: 0,
        };
        unsafe { mem::transmute::<Tensor<'a>, Tensor<'static>>(t) }
    }

    fn from_array_storage<'s, T, D: ndarray::Dimension>(
        arr: &ndarray::Array<T, D>,
        storage: Storage<'s>,
        type_code: usize,
    ) -> Tensor<'s> {
        let type_width = mem::size_of::<T>() as usize;
        Tensor {
            data: storage,
            ctx: TVMContext::default(),
            dtype: DataType {
                code: type_code,
                bits: 8 * type_width,
                lanes: 1,
            },
            size: arr.len(),
            shape: arr.shape().iter().map(|&v| v as i64).collect(),
            strides: Some(arr.strides().iter().map(|&v| v as usize).collect()),
            byte_offset: 0,
        }
    }

    pub(crate) fn as_dltensor(&self, flatten: bool) -> DLTensor {
        assert!(!flatten || self.is_contiguous());
        DLTensor {
            data: unsafe { self.data.as_mut_ptr().offset(self.byte_offset) } as *mut c_void,
            ctx: DLContext::from(&self.ctx),
            ndim: if flatten { 1 } else { self.shape.len() } as i32,
            dtype: DLDataType::from(&self.dtype),
            shape: if flatten {
                &self.size as *const _ as *mut i64
            } else {
                self.shape.as_ptr()
            } as *mut i64,
            strides: if flatten || self.is_contiguous() {
                ptr::null_mut()
            } else {
                self.strides.as_ref().unwrap().as_ptr()
            } as *mut i64,
            byte_offset: 0,
            ..Default::default()
        }
    }
}

/// Conversions to `ndarray::Array` from `Tensor`, if the types match.
macro_rules! impl_ndarray_try_from_tensor {
    ($type:ty, $dtype:expr) => {
        impl<'t> TryFrom<Tensor<'t>> for ndarray::ArrayD<$type> {
            type Error = Error;
            fn try_from(tensor: Tensor) -> Result<ndarray::ArrayD<$type>, Error> {
                ensure!(
                    tensor.dtype == $dtype,
                    "Cannot convert Tensor with dtype {:?} to ndarray",
                    tensor.dtype
                );
                Ok(ndarray::Array::from_shape_vec(
                    tensor
                        .shape
                        .iter()
                        .map(|s| *s as usize)
                        .collect::<Vec<usize>>(),
                    tensor.to_vec::<$type>(),
                )?)
            }
        }
    };
}

macro_rules! make_dtype_const {
    ($name: ident, $code: ident, $bits: expr, $lanes: expr) => {
        pub const $name: DataType = DataType {
            code: $code as usize,
            bits: $bits,
            lanes: $lanes,
        };
    };
}

make_dtype_const!(DTYPE_INT32, DLDataTypeCode_kDLInt, 32, 1);
make_dtype_const!(DTYPE_UINT32, DLDataTypeCode_kDLUInt, 32, 1);
// make_dtype_const!(DTYPE_FLOAT16, DLDataTypeCode_kDLFloat, 16, 1);
make_dtype_const!(DTYPE_FLOAT32, DLDataTypeCode_kDLFloat, 32, 1);
make_dtype_const!(DTYPE_FLOAT64, DLDataTypeCode_kDLFloat, 64, 1);
impl_ndarray_try_from_tensor!(i32, DTYPE_INT32);
impl_ndarray_try_from_tensor!(u32, DTYPE_UINT32);
impl_ndarray_try_from_tensor!(f32, DTYPE_FLOAT32);
impl_ndarray_try_from_tensor!(f64, DTYPE_FLOAT64);

impl<'a, 't> From<&'a Tensor<'t>> for DLTensor {
    fn from(tensor: &'a Tensor<'t>) -> Self {
        Tensor::as_dltensor(tensor, false /* flatten */)
    }
}

impl<'a, 't> From<&'a mut Tensor<'t>> for DLTensor {
    fn from(tensor: &'a mut Tensor<'t>) -> Self {
        Tensor::as_dltensor(tensor, false /* flatten */)
    }
}

impl<'a> From<DLTensor> for Tensor<'a> {
    fn from(dlt: DLTensor) -> Self {
        unsafe {
            let dtype = DataType::from(dlt.dtype);
            let shape = slice::from_raw_parts(dlt.shape, dlt.ndim as usize).to_vec();
            let size = shape.iter().map(|v| *v as usize).product::<usize>() as usize;
            let storage = Storage::from(slice::from_raw_parts(
                dlt.data as *const u8,
                dtype.itemsize() * size,
            ));
            Self {
                data: storage,
                ctx: TVMContext::default(),
                dtype,
                size,
                shape,
                strides: if dlt.strides.is_null() {
                    None
                } else {
                    Some(slice::from_raw_parts_mut(dlt.strides as *mut usize, size).to_vec())
                },
                byte_offset: dlt.byte_offset as isize,
            }
        }
    }
}

/// `From` conversions to `Tensor` for owned or borrowed `ndarray::Array`.
///
/// # Panics
///
/// Panics if the ndarray is not contiguous.
macro_rules! impl_tensor_from_ndarray {
    ($type:ty, $typecode:expr) => {
        impl<D: ndarray::Dimension> From<ndarray::Array<$type, D>> for Tensor<'static> {
            fn from(arr: ndarray::Array<$type, D>) -> Self {
                let storage = Storage::from(arr.as_slice().expect("NDArray must be contiguous"));
                Tensor::from_array_storage(&arr, storage.to_owned(), $typecode as usize)
            }
        }
        impl<'a, D: ndarray::Dimension> From<&'a ndarray::Array<$type, D>> for Tensor<'a> {
            fn from(arr: &'a ndarray::Array<$type, D>) -> Self {
                let storage = Storage::from(arr.as_slice().expect("NDArray must be contiguous"));
                Tensor::from_array_storage(arr, storage, $typecode as usize)
            }
        }
    };
}

impl_tensor_from_ndarray!(f32, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(f64, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(i32, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(i64, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(u32, DLDataTypeCode_kDLUInt);
impl_tensor_from_ndarray!(u64, DLDataTypeCode_kDLUInt);
