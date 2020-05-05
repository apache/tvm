use std::os::raw::c_char;

use crate::ffi::TVMByteArray;

/// A struct holding TVM byte-array.
///
/// ## Example
///
/// ```
/// let v = b"hello";
/// let barr = tvm_sys::ByteArray::from(&v);
/// assert_eq!(barr.len(), v.len());
/// assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
/// ```
pub type ByteArray = TVMByteArray;

impl ByteArray {
    /// Gets the underlying byte-array
    pub fn data(&self) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.data as *const u8, self.size) }
    }

    /// Gets the length of the underlying byte-array
    pub fn len(&self) -> usize {
        self.size
    }

    /// Converts the underlying byte-array to `Vec<u8>`
    pub fn to_vec(&self) -> Vec<u8> {
        self.data().to_vec()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Needs AsRef for Vec
impl<T: AsRef<[u8]>> From<T> for ByteArray {
    fn from(arg: T) -> Self {
        let arg = arg.as_ref();
        ByteArray {
            data: arg.as_ptr() as *const c_char,
            size: arg.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        let v = vec![1u8, 2, 3];
        let barr = ByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.to_vec(), vec![1u8, 2, 3]);
        let v = b"hello";
        let barr = ByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
    }
}
