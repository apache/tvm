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

//! Provides [`TVMByteArray`] used for passing the model parameters
//! (stored as byte-array) to a runtime module.
//!
//! For more detail, please see the example `resnet` in `examples` repository.

use std::os::raw::c_char;

use tvm_common::ffi;

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

impl<'a, T: AsRef<[u8]>> From<T> for TVMByteArray {
    fn from(arg: T) -> Self {
        let arg = arg.as_ref();
        let barr = ffi::TVMByteArray {
            data: arg.as_ptr() as *const c_char,
            size: arg.len(),
        };
        TVMByteArray::new(barr)
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
