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

//! This module implements the [`NDArray`] type for working with *TVM tensors* or
//! coverting from a Rust's ndarray to TVM `NDArray`.
//!
//! One can create an empty NDArray given the shape, context and dtype using [`empty`].
//! To create an NDArray from a mutable buffer in cpu use [`copy_from_buffer`].
//! To copy an NDArray to different context use [`copy_to_ctx`].
//!
//! Given a [`Rust's dynamic ndarray`], one can convert it to TVM NDArray as follows:
//!
//! # Example
//!
//! ```
//! # use tvm_frontend::{NDArray, TVMContext, DataType};
//! # use ndarray::{Array, ArrayD};
//! # use std::str::FromStr;
//! use std::convert::TryFrom;
//!
//! let a = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
//!     .unwrap()
//!     .into_dyn(); // Rust's ndarray
//! let nd = NDArray::from_rust_ndarray(&a, TVMContext::cpu(0), DataType::from_str("float32").unwrap()).unwrap();
//! assert_eq!(nd.shape(), Some(&mut [2, 2][..]));
//! let rnd: ArrayD<f32> = ArrayD::try_from(&nd).unwrap();
//! assert!(rnd.all_close(&a, 1e-8f32));
//! ```
//!
//! [`Rust's dynamic ndarray`]:https://docs.rs/ndarray/0.12.1/ndarray/
//! [`copy_from_buffer`]:struct.NDArray.html#method.copy_from_buffer
//! [`copy_to_ctx`]:struct.NDArray.html#method.copy_to_ctx

use std::{convert::TryFrom, mem, os::raw::c_int, ptr, slice, str::FromStr};

use failure::Error;
use num_traits::Num;
use rust_ndarray::{Array, ArrayD};
use std::convert::TryInto;
use std::ffi::c_void;
use tvm_common::ffi::DLTensor;
use tvm_common::{ffi, TVMType};

use crate::{errors, TVMByteArray, TVMContext};

/// See the [`module-level documentation`](../ndarray/index.html) for more details.
///
/// Wrapper around TVM array handle.
#[derive(Debug)]
pub enum NDArray {
    Borrowed { handle: ffi::TVMArrayHandle },
    Owned { handle: *mut c_void },
}

impl NDArray {
    pub(crate) fn new(handle: ffi::TVMArrayHandle) -> Self {
        NDArray::Borrowed { handle }
    }

    pub(crate) fn from_ndarray_handle(handle: *mut c_void) -> Self {
        NDArray::Owned { handle }
    }

    pub fn as_dltensor(&self) -> &DLTensor {
        unsafe {
            match self {
                NDArray::Borrowed { ref handle } => std::mem::transmute(*handle),
                NDArray::Owned { ref handle } => std::mem::transmute(*handle),
            }
        }
    }

    pub(crate) fn as_raw_dltensor(&self) -> *mut DLTensor {
        unsafe {
            match self {
                NDArray::Borrowed { ref handle } => std::mem::transmute(*handle),
                NDArray::Owned { ref handle } => std::mem::transmute(*handle),
            }
        }
    }

    pub fn is_view(&self) -> bool {
        if let &NDArray::Borrowed { .. } = self {
            true
        } else {
            false
        }
    }

    /// Returns the shape of the NDArray.
    pub fn shape(&self) -> Option<&mut [usize]> {
        let arr = self.as_dltensor();
        if arr.shape.is_null() || arr.data.is_null() {
            return None;
        };
        let slc = unsafe { slice::from_raw_parts_mut(arr.shape as *mut usize, arr.ndim as usize) };
        Some(slc)
    }

    /// Returns the total number of entries of the NDArray.
    pub fn size(&self) -> Option<usize> {
        self.shape().map(|v| v.iter().product())
    }

    /// Returns the context which the NDArray was defined.
    pub fn ctx(&self) -> TVMContext {
        self.as_dltensor().ctx.into()
    }

    /// Returns the type of the entries of the NDArray.
    pub fn dtype(&self) -> TVMType {
        self.as_dltensor().dtype
    }

    /// Returns the number of dimensions of the NDArray.
    pub fn ndim(&self) -> usize {
        self.as_dltensor()
            .ndim
            .try_into()
            .expect("number of dimensions must always be positive")
    }

    /// Returns the strides of the underlying NDArray.
    pub fn strides(&self) -> Option<&[usize]> {
        unsafe {
            let sz = self.ndim() * mem::size_of::<usize>();
            let strides_ptr = self.as_dltensor().strides as *const usize;
            let slc = slice::from_raw_parts(strides_ptr, sz);
            Some(slc)
        }
    }

    /// Shows whether the underlying ndarray is contiguous in memory or not.
    pub fn is_contiguous(&self) -> Result<bool, Error> {
        Ok(match self.strides() {
            None => true,
            Some(strides) => {
                // errors::MissingShapeError in case shape is not determined
                self.shape()
                    .ok_or(errors::MissingShapeError)?
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
        })
    }

    pub fn byte_offset(&self) -> isize {
        self.as_dltensor().byte_offset as isize
    }

    /// Flattens the NDArray to a `Vec` of the same type in cpu.
    ///
    /// ## Example
    ///
    /// ```
    /// # use tvm_frontend::{TVMContext, DataType, NDArray};
    /// # use std::str::FromStr;
    /// let mut shape = [4];
    /// let mut data = vec![1i32, 2, 3, 4];
    /// let ctx = TVMContext::cpu(0);
    /// let mut ndarray = NDArray::empty(&mut shape, ctx, DataType::from_str("int32").unwrap());
    /// ndarray.copy_from_buffer(&mut data);
    /// assert_eq!(ndarray.shape(), Some(&mut shape[..]));
    /// assert_eq!(ndarray.to_vec::<i32>().unwrap(), data);
    /// ```
    pub fn to_vec<T>(&self) -> Result<Vec<T>, Error> {
        ensure!(self.shape().is_some(), errors::EmptyArrayError);
        let earr = NDArray::empty(
            self.shape().ok_or(errors::MissingShapeError)?,
            TVMContext::cpu(0),
            self.dtype(),
        );
        let target = self.copy_to_ndarray(earr)?;
        let arr = target.as_dltensor();
        let sz = self.size().ok_or(errors::MissingShapeError)?;
        let mut v: Vec<T> = Vec::with_capacity(sz * mem::size_of::<T>());
        unsafe {
            v.as_mut_ptr()
                .copy_from_nonoverlapping(arr.data as *const T, sz);
            v.set_len(sz);
        }
        Ok(v)
    }

    /// Converts the NDArray to [`TVMByteArray`].
    pub fn to_bytearray(&self) -> Result<TVMByteArray, Error> {
        let v = self.to_vec::<u8>()?;
        Ok(TVMByteArray::from(v))
    }

    /// Creates an NDArray from a mutable buffer of types i32, u32 or f32 in cpu.
    ///
    /// ## Example
    ///
    /// ```
    /// # use tvm_frontend::{TVMContext, DataType, NDArray};
    /// # use std::str::FromStr;
    /// let shape = &mut [2];
    /// let mut data = vec![1f32, 2.0];
    /// let ctx = TVMContext::cpu(0);
    /// let mut ndarray = NDArray::empty(shape, ctx, DataType::from_str("int32").unwrap());
    /// ndarray.copy_from_buffer(&mut data);
    /// ```
    ///
    /// *Note*: if something goes wrong during the copy, it will panic
    /// from TVM side. See `TVMArrayCopyFromBytes` in `include/tvm/runtime/c_runtime_api.h`.
    pub fn copy_from_buffer<T: Num32>(&mut self, data: &mut [T]) {
        check_call!(ffi::TVMArrayCopyFromBytes(
            self.as_raw_dltensor(),
            data.as_ptr() as *mut _,
            data.len() * mem::size_of::<T>()
        ));
    }

    /// Copies the NDArray to another target NDArray.
    pub fn copy_to_ndarray(&self, target: NDArray) -> Result<NDArray, Error> {
        if self.dtype() != target.dtype() {
            bail!(
                "{}",
                errors::TypeMismatchError {
                    expected: self.dtype().to_string(),
                    actual: target.dtype().to_string(),
                }
            );
        }
        check_call!(ffi::TVMArrayCopyFromTo(
            self.as_raw_dltensor(),
            target.as_raw_dltensor(),
            ptr::null_mut() as ffi::TVMStreamHandle
        ));
        Ok(target)
    }

    /// Copies the NDArray to a target context.
    pub fn copy_to_ctx(&self, target: &TVMContext) -> Result<NDArray, Error> {
        let tmp = NDArray::empty(
            self.shape().ok_or(errors::MissingShapeError)?,
            *target,
            self.dtype(),
        );
        let copy = self.copy_to_ndarray(tmp)?;
        Ok(copy)
    }

    /// Converts a Rust's ndarray to TVM NDArray.
    pub fn from_rust_ndarray<T: Num32 + Copy>(
        rnd: &ArrayD<T>,
        ctx: TVMContext,
        dtype: TVMType,
    ) -> Result<Self, Error> {
        let shape = rnd.shape().to_vec();
        let mut nd = NDArray::empty(&shape, ctx, dtype);
        let mut buf = Array::from_iter(rnd.into_iter().map(|&v| v as T));
        nd.copy_from_buffer(
            buf.as_slice_mut()
                .expect("Array from iter must be contiguous."),
        );
        Ok(nd)
    }

    /// Allocates and creates an empty NDArray given the shape, context and dtype.
    pub fn empty(shape: &[usize], ctx: TVMContext, dtype: TVMType) -> NDArray {
        let mut handle = ptr::null_mut() as ffi::TVMArrayHandle;
        check_call!(ffi::TVMArrayAlloc(
            shape.as_ptr() as *const i64,
            shape.len() as c_int,
            i32::from(dtype.code) as c_int,
            i32::from(dtype.bits) as c_int,
            i32::from(dtype.lanes) as c_int,
            ctx.device_type.0 as c_int,
            ctx.device_id as c_int,
            &mut handle as *mut _,
        ));
        NDArray::Borrowed { handle: handle }
    }
}

macro_rules! impl_from_ndarray_rustndarray {
    ($type:ty, $type_name:tt) => {
        impl<'a> TryFrom<&'a NDArray> for ArrayD<$type> {
            type Error = Error;
            fn try_from(nd: &NDArray) -> Result<ArrayD<$type>, Self::Error> {
                ensure!(nd.shape().is_some(), errors::MissingShapeError);
                assert_eq!(nd.dtype(), TVMType::from_str($type_name)?, "Type mismatch");
                Ok(Array::from_shape_vec(
                    &*nd.shape().ok_or(errors::MissingShapeError)?,
                    nd.to_vec::<$type>()?,
                )?)
            }
        }

        impl<'a> TryFrom<&'a mut NDArray> for ArrayD<$type> {
            type Error = Error;
            fn try_from(nd: &mut NDArray) -> Result<ArrayD<$type>, Self::Error> {
                ensure!(nd.shape().is_some(), errors::MissingShapeError);
                assert_eq!(nd.dtype(), TVMType::from_str($type_name)?, "Type mismatch");
                Ok(Array::from_shape_vec(
                    &*nd.shape().ok_or(errors::MissingShapeError)?,
                    nd.to_vec::<$type>()?,
                )?)
            }
        }
    };
}

impl_from_ndarray_rustndarray!(i32, "int");
impl_from_ndarray_rustndarray!(u32, "uint");
impl_from_ndarray_rustndarray!(f32, "float");

impl Drop for NDArray {
    fn drop(&mut self) {
        if let &mut NDArray::Owned { .. } = self {
            check_call!(ffi::TVMArrayFree(self.as_raw_dltensor()));
        }
    }
}

mod sealed {
    /// Private trait to prevent other traits from being implemeneted in downstream crates.
    pub trait Sealed {}
}

/// A trait for the supported 32-bits numerical types in frontend.
pub trait Num32: Num + sealed::Sealed {
    const BITS: u8 = 32;
}

macro_rules! impl_num32 {
    ($($type:ty),+) => {
        $(
            impl sealed::Sealed for $type {}
            impl Num32 for $type {}
        )+
    };
}

impl_num32!(i32, u32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let shape = &mut [1, 2, 3];
        let ctx = TVMContext::cpu(0);
        let ndarray = NDArray::empty(shape, ctx, TVMType::from_str("int32").unwrap());
        assert_eq!(ndarray.shape().unwrap(), shape);
        assert_eq!(
            ndarray.size().unwrap(),
            shape.to_vec().into_iter().product()
        );
        assert_eq!(ndarray.ndim(), 3);
        assert!(ndarray.strides().is_none());
        assert_eq!(ndarray.byte_offset(), 0);
    }

    #[test]
    fn copy() {
        let shape = &mut [4];
        let mut data = vec![1i32, 2, 3, 4];
        let ctx = TVMContext::cpu(0);
        let mut ndarray = NDArray::empty(shape, ctx, TVMType::from_str("int32").unwrap());
        assert!(ndarray.to_vec::<i32>().is_ok());
        ndarray.copy_from_buffer(&mut data);
        assert_eq!(ndarray.shape().unwrap(), shape);
        assert_eq!(ndarray.to_vec::<i32>().unwrap(), data);
        assert_eq!(ndarray.ndim(), 1);
        assert!(ndarray.is_contiguous().is_ok());
        assert_eq!(ndarray.byte_offset(), 0);
        let shape = vec![4];
        let e = NDArray::empty(
            &shape,
            TVMContext::cpu(0),
            TVMType::from_str("int32").unwrap(),
        );
        let nd = ndarray.copy_to_ndarray(e);
        assert!(nd.is_ok());
        assert_eq!(nd.unwrap().to_vec::<i32>().unwrap(), data);
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err`")]
    fn copy_wrong_dtype() {
        let shape = vec![4];
        let mut data = vec![1f32, 2., 3., 4.];
        let ctx = TVMContext::cpu(0);
        let mut nd_float = NDArray::empty(&shape, ctx, TVMType::from_str("float32").unwrap());
        nd_float.copy_from_buffer(&mut data);
        let empty_int = NDArray::empty(&shape, ctx, TVMType::from_str("int32").unwrap());
        nd_float.copy_to_ndarray(empty_int).unwrap();
    }

    #[test]
    fn rust_ndarray() {
        let a = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
            .unwrap()
            .into_dyn();
        let nd = NDArray::from_rust_ndarray(
            &a,
            TVMContext::cpu(0),
            TVMType::from_str("float32").unwrap(),
        )
        .unwrap();
        assert_eq!(nd.shape().unwrap(), &mut [2, 2]);
        let rnd: ArrayD<f32> = ArrayD::try_from(&nd).unwrap();
        assert!(rnd.all_close(&a, 1e-8f32));
    }
}
