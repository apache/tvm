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
//! One can create an empty NDArray given the shape, device and dtype using [`empty`].
//! To create an NDArray from a mutable buffer in cpu use [`copy_from_buffer`].
//! To copy an NDArray to different device use [`copy_to_device`].
//!
//! Given a [`Rust's dynamic ndarray`], one can convert it to TVM NDArray as follows:
//!
//! # Example
//!
//! ```
//! # use tvm_rt::{NDArray, DataType, Device};
//! # use ndarray::{Array, ArrayD};
//! # use std::str::FromStr;
//! use std::convert::TryFrom;
//!
//! let a = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
//!     .unwrap()
//!     .into_dyn(); // Rust's ndarray
//! let nd = NDArray::from_rust_ndarray(&a, Device::cpu(0), DataType::from_str("float32").unwrap()).unwrap();
//! assert_eq!(nd.shape(), &[2, 2]);
//! let rnd: ArrayD<f32> = ArrayD::try_from(&nd).unwrap();
//! assert!(rnd.all_close(&a, 1e-8f32));
//! ```
//!
//! [`Rust's dynamic ndarray`]:https://docs.rs/ndarray/0.12.1/ndarray/
//! [`copy_from_buffer`]:struct.NDArray.html#method.copy_from_buffer
//! [`copy_to_device`]:struct.NDArray.html#method.copy_to_device

use std::ffi::c_void;
use std::{borrow::Cow, convert::TryInto};
use std::{convert::TryFrom, mem, os::raw::c_int, ptr, slice, str::FromStr};

use mem::size_of;
use tvm_macros::Object;
use tvm_sys::ffi::DLTensor;
use tvm_sys::{ffi, ByteArray, DataType, Device};

use ndarray::{Array, ArrayD};
use num_traits::Num;

use crate::errors::NDArrayError;

use crate::object::{Object, ObjectPtr, ObjectRef};

/// See the [`module-level documentation`](../ndarray/index.html) for more details.
#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "NDArray"]
#[type_key = "runtime.NDArray"]
pub struct NDArrayContainer {
    base: Object,
    // Container Base
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    shape: ObjectRef,
}

impl NDArrayContainer {
    pub(crate) fn from_raw(handle: ffi::TVMArrayHandle) -> Option<ObjectPtr<Self>> {
        let base_offset = memoffset::offset_of!(NDArrayContainer, dl_tensor) as isize;
        let base_ptr = unsafe { (handle as *mut i8).offset(-base_offset) };
        let object_ptr = ObjectPtr::from_raw(base_ptr.cast());
        object_ptr.map(|ptr| {
            ptr.downcast::<NDArrayContainer>()
                .expect("we know this is an NDArray container")
        })
    }

    pub fn leak<'a>(object_ptr: ObjectPtr<NDArrayContainer>) -> &'a mut NDArrayContainer
    where
        NDArrayContainer: 'a,
    {
        let base_offset = memoffset::offset_of!(NDArrayContainer, dl_tensor) as isize;
        unsafe {
            &mut *std::mem::ManuallyDrop::new(object_ptr)
                .ptr
                .as_ptr()
                .cast::<u8>()
                .offset(base_offset)
                .cast::<NDArrayContainer>()
        }
    }

    pub fn as_mut_ptr<'a>(object_ptr: &ObjectPtr<NDArrayContainer>) -> *mut NDArrayContainer
    where
        NDArrayContainer: 'a,
    {
        let base_offset = memoffset::offset_of!(NDArrayContainer, dl_tensor) as isize;
        unsafe {
            object_ptr
                .ptr
                .as_ptr()
                .cast::<u8>()
                .offset(base_offset)
                .cast::<NDArrayContainer>()
        }
    }
}

fn cow_usize<'a>(slice: &[i64]) -> Cow<'a, [usize]> {
    if std::mem::size_of::<usize>() == 64 {
        debug_assert!(slice.iter().all(|&x| x >= 0));
        let shape: &[usize] = unsafe { std::mem::transmute(slice) };
        Cow::Borrowed(shape)
    } else {
        let shape: Vec<usize> = slice
            .iter()
            .map(|&x| usize::try_from(x).unwrap_or_else(|_| panic!("Cannot fit into usize: {}", x)))
            .collect();
        Cow::Owned(shape)
    }
}

impl NDArray {
    pub(crate) fn _from_raw(handle: ffi::TVMArrayHandle) -> Self {
        let ptr = NDArrayContainer::from_raw(handle);
        NDArray(ptr)
    }

    // I think these should be marked as unsafe functions? projecting a reference is bad news.
    pub fn as_dltensor(&self) -> &DLTensor {
        &self.dl_tensor
    }

    pub(crate) fn as_raw_dltensor(&self) -> *mut DLTensor {
        unsafe { std::mem::transmute(self.as_dltensor()) }
    }

    pub fn is_view(&self) -> bool {
        false
    }

    /// Returns the shape of the NDArray.
    pub fn shape(&self) -> &[i64] {
        let arr = self.as_dltensor();
        if arr.shape.is_null() || arr.data.is_null() {
            &[]
        } else {
            unsafe { slice::from_raw_parts(arr.shape, self.ndim()) }
        }
    }

    /// Returns the shape of the NDArray as a &[usize]
    ///
    /// On 64-bit platforms, this is zero-cost and uses the shape from the DLTensor.
    /// On other platforms, this copies into a buffer.
    pub fn shape_usize(&self) -> Cow<[usize]> {
        cow_usize(self.shape())
    }

    /// Returns the strides of the underlying NDArray.
    pub fn strides(&self) -> Option<&[i64]> {
        let arr = self.as_dltensor();
        if arr.strides.is_null() {
            None
        } else {
            Some(unsafe { slice::from_raw_parts(arr.strides, self.ndim()) })
        }
    }

    /// Returns the strides of the NDArray as a &[usize]
    ///
    /// On 64-bit platforms, this is zero-cost and uses the strides from the DLTensor.
    /// On other platforms, this copies into a buffer.
    pub fn strides_usize(&self) -> Option<Cow<[usize]>> {
        self.strides().map(cow_usize)
    }

    /// Returns true if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.as_dltensor().data.is_null()
    }

    /// Returns the total number of entries of the NDArray.
    pub fn len(&self) -> usize {
        let len: i64 = self.shape().iter().product();
        usize::try_from(len).unwrap_or_else(|_| panic!("bad len: {}", len))
    }

    /// Returns the total bytes taken up by the data.
    /// This is equal to `nd.len() * nd.dtype().itemsize()`
    pub fn size(&self) -> usize {
        self.len() * self.dtype().itemsize()
    }

    /// Returns the device which the NDArray was defined.
    pub fn device(&self) -> Device {
        self.as_dltensor().device.into()
    }

    /// Returns the type of the entries of the NDArray.
    pub fn dtype(&self) -> DataType {
        self.as_dltensor().dtype.into()
    }

    /// Returns the number of dimensions of the NDArray.
    pub fn ndim(&self) -> usize {
        self.as_dltensor()
            .ndim
            .try_into()
            .expect("number of dimensions must always be positive")
    }

    /// Shows whether the underlying ndarray is contiguous in memory or not.
    pub fn is_contiguous(&self) -> bool {
        match self.strides() {
            None => true,
            Some(strides) => {
                // NDArrayError::MissingShape in case shape is not determined
                self.shape()
                    .iter()
                    .zip(strides)
                    .rfold(
                        (true, 1),
                        |(is_contig, expected_stride), (shape, stride)| {
                            (
                                is_contig && *stride == expected_stride,
                                expected_stride * shape,
                            )
                        },
                    )
                    .0
            }
        }
    }

    pub fn byte_offset(&self) -> isize {
        self.as_dltensor().byte_offset as isize
    }

    /// Flattens the NDArray to a `Vec` of the same type in cpu.
    ///
    /// ## Example
    ///
    /// ```
    /// # use tvm_rt::{Device, DataType, NDArray};
    /// # use std::str::FromStr;
    /// let mut shape = [4];
    /// let mut data = vec![1i32, 2, 3, 4];
    /// let dev = Device::cpu(0);
    /// let mut ndarray = NDArray::empty(&mut shape, dev, DataType::from_str("int32").unwrap());
    /// ndarray.copy_from_buffer(&mut data);
    /// assert_eq!(ndarray.shape(), shape);
    /// assert_eq!(ndarray.to_vec::<i32>().unwrap(), data);
    /// ```
    pub fn to_vec<T>(&self) -> Result<Vec<T>, NDArrayError> {
        let n = self.size() / size_of::<T>();
        let mut vec: Vec<T> = Vec::with_capacity(n);

        let ptr = vec.as_mut_ptr();
        let slice = unsafe { slice::from_raw_parts_mut(ptr, n) };
        self.copy_to_buffer(slice);

        unsafe { vec.set_len(n) };
        Ok(vec)
    }

    /// Converts the NDArray to [`ByteArray`].
    pub fn to_bytearray(&self) -> Result<ByteArray, NDArrayError> {
        let v = self.to_vec::<u8>()?;
        Ok(ByteArray::from(v))
    }

    /// Creates an NDArray from a mutable buffer of types i32, u32 or f32 in cpu.
    ///
    /// ## Example
    ///
    /// ```
    /// # use tvm_rt::{Device, DataType, NDArray};
    /// # use std::str::FromStr;
    /// let shape = &mut [2];
    /// let mut data = vec![1f32, 2.0];
    /// let dev = Device::cpu(0);
    /// let mut ndarray = NDArray::empty(shape, dev, DataType::from_str("int32").unwrap());
    /// ndarray.copy_from_buffer(&mut data);
    /// ```
    ///
    /// *Note*: if something goes wrong during the copy, it will panic
    /// from TVM side. See `TVMArrayCopyFromBytes` in `include/tvm/runtime/c_runtime_api.h`.
    pub fn copy_from_buffer<T: Num32>(&mut self, data: &[T]) {
        check_call!(ffi::TVMArrayCopyFromBytes(
            self.as_raw_dltensor(),
            data.as_ptr() as *mut _,
            (data.len() * mem::size_of::<T>()) as _,
        ));
    }

    pub fn copy_to_buffer<T>(&self, data: &mut [T]) {
        assert_eq!(self.size(), data.len() * size_of::<T>());
        check_call!(ffi::TVMArrayCopyToBytes(
            self.as_raw_dltensor(),
            data.as_ptr() as *mut _,
            self.size() as _,
        ));
    }

    pub fn fill_from_iter<T, I>(&mut self, iter: I)
    where
        T: Num32,
        I: ExactSizeIterator<Item = T>,
    {
        assert!(self.is_contiguous());
        assert_eq!(self.size(), size_of::<T>() * iter.len());
        let mut ptr: *mut T = self.as_dltensor().data.cast();
        iter.for_each(|x| unsafe {
            ptr.write(x);
            ptr = ptr.add(1);
        })
    }

    /// Copies the NDArray to another target NDArray.
    pub fn copy_to_ndarray(&self, target: NDArray) -> Result<NDArray, NDArrayError> {
        if self.dtype() != target.dtype() {
            return Err(NDArrayError::DataTypeMismatch {
                expected: self.dtype(),
                actual: target.dtype(),
            });
        }

        check_call!(ffi::TVMArrayCopyFromTo(
            self.as_raw_dltensor(),
            target.as_raw_dltensor(),
            ptr::null_mut() as ffi::TVMStreamHandle
        ));

        Ok(target)
    }

    /// Copies the NDArray to a target device.
    pub fn copy_to_device(&self, target: &Device) -> Result<NDArray, NDArrayError> {
        let tmp = NDArray::empty(self.shape(), *target, self.dtype());
        let copy = self.copy_to_ndarray(tmp)?;
        Ok(copy)
    }

    /// Converts a Rust's ndarray to TVM NDArray.
    pub fn from_rust_ndarray<T: Num32 + Copy>(
        input_nd: &ArrayD<T>,
        dev: Device,
        dtype: DataType,
    ) -> Result<Self, NDArrayError> {
        let shape: Vec<i64> = input_nd.shape().iter().map(|&x| x as i64).collect();
        let mut nd = NDArray::empty(&shape, dev, dtype);
        nd.fill_from_iter(input_nd.iter().copied());
        Ok(nd)
    }

    /// Allocates and creates an empty NDArray given the shape, device and dtype.
    pub fn empty(shape: &[i64], dev: Device, dtype: DataType) -> NDArray {
        let mut handle = ptr::null_mut() as ffi::TVMArrayHandle;
        let dtype: tvm_sys::ffi::DLDataType = dtype.into();
        check_call!(ffi::TVMArrayAlloc(
            shape.as_ptr(),
            shape.len() as c_int,
            i32::from(dtype.code) as c_int,
            i32::from(dtype.bits) as c_int,
            i32::from(dtype.lanes) as c_int,
            dev.device_type as c_int,
            dev.device_id as c_int,
            &mut handle as *mut _,
        ));
        let ptr = NDArrayContainer::from_raw(handle)
            .map(|o| o.downcast().expect("this should never fail"));
        NDArray(ptr)
    }

    pub fn zeroed(self) -> NDArray {
        unsafe {
            let dltensor = self.as_raw_dltensor();
            let bytes_ptr: *mut u8 = std::mem::transmute((*dltensor).data);
            println!("size {}", self.size());
            std::ptr::write_bytes(bytes_ptr, 0, self.size());
            self
        }
    }
}

macro_rules! impl_from_ndarray_rustndarray {
    ($type:ty, $type_name:tt) => {
        impl<'a> TryFrom<&'a NDArray> for ArrayD<$type> {
            type Error = NDArrayError;

            fn try_from(nd: &NDArray) -> Result<ArrayD<$type>, Self::Error> {
                assert_eq!(nd.dtype(), DataType::from_str($type_name)?, "Type mismatch");
                Ok(Array::from_shape_vec(
                    &*nd.shape_usize(),
                    nd.to_vec::<$type>()?,
                )?)
            }
        }

        impl<'a> TryFrom<&'a mut NDArray> for ArrayD<$type> {
            type Error = NDArrayError;

            fn try_from(nd: &mut NDArray) -> Result<ArrayD<$type>, Self::Error> {
                assert_eq!(nd.dtype(), DataType::from_str($type_name)?, "Type mismatch");
                Ok(Array::from_shape_vec(
                    &*nd.shape_usize(),
                    nd.to_vec::<$type>()?,
                )?)
            }
        }
    };
}

impl_from_ndarray_rustndarray!(i32, "int");
impl_from_ndarray_rustndarray!(u32, "uint");
impl_from_ndarray_rustndarray!(f32, "float");

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
        let shape = &[1, 2, 3];
        let dev = Device::cpu(0);
        println!("before empty");
        let ndarray = NDArray::empty(shape, dev, DataType::from_str("int32").unwrap());
        println!("after empty");
        assert_eq!(ndarray.shape(), shape);
        assert_eq!(ndarray.len(), shape.iter().product::<i64>() as usize);
        assert_eq!(ndarray.ndim(), 3);
        assert!(ndarray.strides().is_none());
        assert_eq!(ndarray.byte_offset(), 0);
    }

    #[test]
    fn copy() {
        let shape = &[4];
        let data = vec![1i32, 2, 3, 4];
        let dev = Device::cpu(0);
        let mut ndarray = NDArray::empty(shape, dev, DataType::int(32, 1)).zeroed();
        assert_eq!(ndarray.to_vec::<i32>().unwrap(), vec![0, 0, 0, 0]);
        ndarray.copy_from_buffer(&data);
        assert_eq!(ndarray.shape(), shape);
        assert_eq!(ndarray.to_vec::<i32>().unwrap(), data);
        assert_eq!(ndarray.ndim(), 1);
        assert!(ndarray.is_contiguous());
        assert_eq!(ndarray.byte_offset(), 0);
        let shape = vec![4];
        let e = NDArray::empty(&shape, Device::cpu(0), DataType::from_str("int32").unwrap());
        let nd = ndarray.copy_to_ndarray(e);
        assert!(nd.is_ok());
        assert_eq!(nd.unwrap().to_vec::<i32>().unwrap(), data);
    }

    /// This occasionally panics on macOS: https://github.com/rust-lang/rust/issues/71397
    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err`")]
    fn copy_wrong_dtype() {
        let shape = vec![4];
        let mut data = vec![1f32, 2., 3., 4.];
        let dev = Device::cpu(0);
        let mut nd_float = NDArray::empty(&shape, dev, DataType::from_str("float32").unwrap());
        nd_float.copy_from_buffer(&mut data);
        let empty_int = NDArray::empty(&shape, dev, DataType::from_str("int32").unwrap());
        nd_float.copy_to_ndarray(empty_int).unwrap();
    }

    #[test]
    fn rust_ndarray() {
        let a = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
            .unwrap()
            .into_dyn();
        let nd =
            NDArray::from_rust_ndarray(&a, Device::cpu(0), DataType::from_str("float32").unwrap())
                .unwrap();
        assert_eq!(nd.shape(), &[2, 2]);
        let rnd: ArrayD<f32> = ArrayD::try_from(&nd).unwrap();
        assert!(rnd.all_close(&a, 1e-8f32));
    }
}
