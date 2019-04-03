//! Provides [`TVMByteArray`] used for passing the model parameters
//! (stored as byte-array) to a runtime module.
//!
//! For more detail, please see the example `resnet` in `examples` repository.

use std::os::raw::{c_char, c_void};

use tvm_common::{ffi, TVMArgValue};

/// A struct holding TVM byte-array.
///
/// ## Example
///
/// ```
/// let v = b"hello".to_vec();
/// let barr = TVMByteArray::from(&v);
/// assert_eq!(barr.len(), v.len());
/// assert_eq!(barr.data(), vec![104i8, 101, 108, 108, 111]);
/// ```
#[derive(Debug, Clone)]
pub struct TVMByteArray {
    pub(crate) inner: ffi::TVMByteArray,
}

impl TVMByteArray {
    pub(crate) fn new(barr: ffi::TVMByteArray) -> TVMByteArray {
        TVMByteArray { inner: barr }
    }

    /// Gets the length of the underlying byte-array
    pub fn len(&self) -> usize {
        self.inner.size
    }

    /// Gets the underlying byte-array as `Vec<i8>`
    pub fn data(&self) -> Vec<i8> {
        unsafe {
            let sz = self.len();
            let mut ret_buf = Vec::with_capacity(sz);
            ret_buf.set_len(sz);
            self.inner.data.copy_to(ret_buf.as_mut_ptr(), sz);
            ret_buf
        }
    }
}

impl<'a> From<&'a Vec<u8>> for TVMByteArray {
    fn from(arg: &Vec<u8>) -> Self {
        let barr = ffi::TVMByteArray {
            data: arg.as_ptr() as *const c_char,
            size: arg.len(),
        };
        TVMByteArray::new(barr)
    }
}

impl<'a> From<&TVMByteArray> for TVMArgValue<'a> {
    fn from(arr: &TVMByteArray) -> Self {
        Self {
            value: ffi::TVMValue {
                v_handle: &arr.inner as *const ffi::TVMByteArray as *const c_void as *mut c_void,
            },
            type_code: ffi::TVMTypeCode_kBytes as i64,
            _lifetime: std::marker::PhantomData,
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
        assert_eq!(barr.data(), vec![1i8, 2, 3]);
        let v = b"hello".to_vec();
        let barr = TVMByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), vec![104i8, 101, 108, 108, 111]);
    }
}
