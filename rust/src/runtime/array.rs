use std::{
  any::TypeId,
  convert::TryFrom,
  mem,
  os::raw::{c_int, c_void},
  ptr, slice,
};

use ndarray;

use super::allocator::Allocation;
use errors::*;
use ffi::runtime::{
  DLContext, DLDataType, DLDataTypeCode_kDLFloat, DLDataTypeCode_kDLInt, DLDataTypeCode_kDLUInt,
  DLDeviceType_kDLCPU, DLTensor,
};

/// A `Storage` is a container which holds `Tensor` data.
#[derive(PartialEq)]
pub enum Storage<'a> {
  /// A `Storage` which owns its contained bytes.
  Owned(Allocation),

  /// A view of an existing `Storage`.
  View(&'a mut [u8], usize), // ptr, align
}

impl<'a> Storage<'a> {
  pub fn new(size: usize, align: Option<usize>) -> Result<Storage<'static>> {
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
        .copy_from_nonoverlapping(self.as_ptr(), self.size())
    }
    s
  }
}

impl<'a, T> From<&'a [T]> for Storage<'a> {
  fn from(data: &'a [T]) -> Self {
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
///
/// let mut a_nd: ndarray::Array = ndarray::Array::from_vec(vec![1f32, 2., 3., 4.]);
/// let mut a: Tensor = a_nd.into();
/// let mut a_dl: DLTensor = (&mut t).into();
/// call_packed!(tvm_fn, &mut a_dl);
///
/// // Array -> Tensor is mostly useful when post-processing TVM graph outputs.
/// let mut a_nd = ndarray::Array::try_from(&a).unwrap();
/// ```
#[derive(PartialEq)]
pub struct Tensor<'a> {
  /// The bytes which contain the data this `Tensor` represents.
  pub(super) data: Storage<'a>,
  pub(super) ctx: TVMContext,
  pub(super) dtype: DataType,
  pub(super) shape: Vec<i64>, // not usize because `typedef int64_t tvm_index_t` in c_runtime_api.h
  /// The `Tensor` strides. Can be `None` if the `Tensor` is contiguous.
  pub(super) strides: Option<Vec<usize>>,
  pub(super) byte_offset: isize,
  pub(super) size: usize,
}

unsafe impl<'a> Send for Tensor<'a> {}

impl<'a> Tensor<'a> {
  pub fn shape(&self) -> Vec<i64> {
    self.shape.clone()
  }

  /// Returns the data of this `Tensor` as a `Vec`.
  ///
  /// # Panics
  ///
  /// Panics if the `Tensor` is not contiguous or does not contain elements of type `T`.
  pub fn to_vec<T: 'static>(&self) -> Vec<T> {
    assert!(self.is_contiguous());
    assert!(self.dtype.is_type::<T>());
    let mut vec: Vec<T> = Vec::with_capacity(self.size * self.dtype.itemsize());
    unsafe {
      vec.as_mut_ptr().copy_from_nonoverlapping(
        self.data.as_ptr().offset(self.byte_offset) as *const T,
        self.size,
      );
      vec.set_len(self.size);
    }
    vec
  }

  /// Returns `true` iff this `Tensor` is represented by a contiguous region of memory.
  pub fn is_contiguous(&self) -> bool {
    match self.strides {
      None => true,
      Some(ref strides) => {
        // check that stride for each dimension is the product of all trailing dimensons' shapes
        self
          .shape
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
          ).0
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
      self
        .data
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
      ctx: self.ctx.clone(),
      dtype: self.dtype.clone(),
      size: self.size.clone(),
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
      strides: Some(arr.strides().into_iter().map(|&v| v as usize).collect()),
      byte_offset: 0,
    }
  }
}

/// Conversions to `ndarray::Array` from `Tensor`, if the types match.
macro_rules! impl_ndarray_try_from_tensor {
  ($type:ty, $dtype:expr) => {
    impl<'a, 't> TryFrom<&'a Tensor<'t>> for ndarray::ArrayD<$type> {
      type Error = Error;
      fn try_from(tensor: &'a Tensor) -> Result<ndarray::ArrayD<$type>> {
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

impl_ndarray_try_from_tensor!(i32, DTYPE_INT32);
impl_ndarray_try_from_tensor!(u32, DTYPE_UINT32);
impl_ndarray_try_from_tensor!(f32, DTYPE_FLOAT32);
impl_ndarray_try_from_tensor!(f64, DTYPE_FLOAT64);

impl DLTensor {
  pub(super) fn from_tensor<'a>(tensor: &'a Tensor, flatten: bool) -> Self {
    assert!(!flatten || tensor.is_contiguous());
    Self {
      data: unsafe { tensor.data.as_mut_ptr().offset(tensor.byte_offset) } as *mut c_void,
      ctx: DLContext::from(&tensor.ctx),
      ndim: if flatten { 1 } else { tensor.shape.len() } as i32,
      dtype: DLDataType::from(&tensor.dtype),
      shape: if flatten {
        &tensor.size as *const _ as *mut i64
      } else {
        tensor.shape.as_ptr()
      } as *mut i64,
      strides: if flatten || tensor.is_contiguous() {
        ptr::null_mut()
      } else {
        tensor.strides.as_ref().unwrap().as_ptr()
      } as *mut i64,
      byte_offset: 0,
    }
  }
}

impl<'a, 't> From<&'a Tensor<'t>> for DLTensor {
  fn from(tensor: &'a Tensor<'t>) -> Self {
    DLTensor::from_tensor(tensor, false /* flatten */)
  }
}

impl<'a, 't> From<&'a mut Tensor<'t>> for DLTensor {
  fn from(tensor: &'a mut Tensor<'t>) -> Self {
    DLTensor::from_tensor(tensor, false /* flatten */)
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataType {
  pub(super) code: usize,
  pub(super) bits: usize,
  pub(super) lanes: usize,
}

impl DataType {
  /// Returns the number of bytes occupied by an element of this `DataType`.
  fn itemsize(&self) -> usize {
    (self.bits * self.lanes) >> 3
  }

  /// Returns whether this `DataType` represents primitive type `T`.
  fn is_type<T: 'static>(&self) -> bool {
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

macro_rules! make_dtype_const {
  ($name: ident, $code: ident, $bits: expr, $lanes: expr) => {
    const $name: DataType = DataType {
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

impl Default for DLContext {
  fn default() -> Self {
    DLContext {
      device_type: DLDeviceType_kDLCPU,
      device_id: 0,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TVMContext {
  pub(super) device_type: usize,
  pub(super) device_id: usize,
}

impl<'a> From<&'a TVMContext> for DLContext {
  fn from(ctx: &'a TVMContext) -> Self {
    Self {
      device_type: ctx.device_type as u32,
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

/// `From` conversions to `Tensor` for owned or borrowed `ndarray::Array`.
///
/// # Panics
///
/// Panics if the ndarray is not contiguous.
macro_rules! impl_tensor_from_ndarray {
  ($type:ty, $typecode:expr) => {
    impl<D: ndarray::Dimension> From<ndarray::Array<$type, D>> for Tensor<'static> {
      fn from(arr: ndarray::Array<$type, D>) -> Self {
        assert!(arr.is_standard_layout(), "Array must be contiguous.");
        let size = arr.len() * mem::size_of::<$type>() as usize;
        let storage =
          Storage::from(unsafe { slice::from_raw_parts(arr.as_ptr() as *const u8, size) });
        Tensor::from_array_storage(&arr, storage, $typecode as usize)
      }
    }
    impl<'a, D: ndarray::Dimension> From<&'a ndarray::Array<$type, D>> for Tensor<'a> {
      fn from(arr: &'a ndarray::Array<$type, D>) -> Self {
        assert!(arr.is_standard_layout(), "Array must be contiguous.");
        Tensor::from_array_storage(
          arr,
          Storage::from(arr.as_slice().unwrap()),
          $typecode as usize,
        )
      }
    }
  };
}

/// `From` conversions to `DLTensor` for `ndarray::Array`.
/// Takes a reference to the `ndarray` since `DLTensor` is not owned.
macro_rules! impl_dltensor_from_ndarray {
  ($type:ty, $typecode:expr) => {
    impl<'a, D: ndarray::Dimension> From<&'a mut ndarray::Array<$type, D>> for DLTensor {
      fn from(arr: &'a mut ndarray::Array<$type, D>) -> Self {
        DLTensor {
          data: arr.as_mut_ptr() as *mut c_void,
          ctx: DLContext::default(),
          ndim: arr.ndim() as c_int,
          dtype: DLDataType {
            code: $typecode as u8,
            bits: 8 * mem::size_of::<$type>() as u8,
            lanes: 1,
          },
          shape: arr.shape().as_ptr() as *const i64 as *mut i64,
          strides: arr.strides().as_ptr() as *const isize as *mut i64,
          byte_offset: 0,
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

impl_tensor_from_ndarray!(f32, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(f64, DLDataTypeCode_kDLFloat);
impl_tensor_from_ndarray!(i32, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(i64, DLDataTypeCode_kDLInt);
impl_tensor_from_ndarray!(u32, DLDataTypeCode_kDLUInt);
impl_tensor_from_ndarray!(u64, DLDataTypeCode_kDLUInt);
