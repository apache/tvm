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
use std::convert::TryFrom;

use crate::errors::ValueDowncastError;
use crate::ffi::{TVMByteArray, TVMByteArrayFree};
use crate::{ArgValue, RetValue};

/// A newtype wrapping a raw TVM byte-array.
///
/// ## Example
///
/// ```
/// let v = b"hello";
/// let barr = tvm_sys::ByteArray::from(&v);
/// assert_eq!(barr.len(), v.len());
/// assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
/// ```
pub enum ByteArray {
    Rust(TVMByteArray),
    External(TVMByteArray),
}

impl Drop for ByteArray {
    fn drop(&mut self) {
        match self {
            ByteArray::Rust(bytes) => {
                let ptr = bytes.data;
                let len = bytes.size as _;
                let cap = bytes.size as _;
                let data: Vec<u8> = unsafe { Vec::from_raw_parts(ptr as _, len, cap) };
                drop(data);
            }
            ByteArray::External(byte_array) => unsafe {
                if TVMByteArrayFree(byte_array as _) != 0 {
                    panic!("error");
                }
            },
        }
    }
}

impl ByteArray {
    /// Gets the underlying byte-array
    pub fn data(&self) -> &[u8] {
        match self {
            ByteArray::Rust(byte_array) | ByteArray::External(byte_array) => unsafe {
                std::slice::from_raw_parts(byte_array.data as *const u8, byte_array.size as _)
            },
        }
    }

    /// Gets the length of the underlying byte-array
    pub fn len(&self) -> usize {
        match self {
            ByteArray::Rust(byte_array) | ByteArray::External(byte_array) => byte_array.size as _,
        }
    }

    /// Converts the underlying byte-array to `Vec<u8>`
    pub fn to_vec(&self) -> Vec<u8> {
        self.data().to_vec()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Into<Vec<u8>>> From<T> for ByteArray {
    fn from(arg: T) -> Self {
        let mut incoming_bytes: Vec<u8> = arg.into();
        let mut bytes = Vec::with_capacity(incoming_bytes.len());
        bytes.append(&mut incoming_bytes);

        let mut bytes = std::mem::ManuallyDrop::new(bytes);
        let ptr = bytes.as_mut_ptr();
        assert_eq!(bytes.len(), bytes.capacity());
        ByteArray::Rust(TVMByteArray {
            data: ptr as _,
            size: bytes.len() as _,
        })
    }
}

impl<'a> From<&'a ByteArray> for ArgValue<'a> {
    fn from(val: &'a ByteArray) -> ArgValue<'a> {
        match val {
            ByteArray::Rust(byte_array) | ByteArray::External(byte_array) => {
                ArgValue::Bytes(byte_array)
            }
        }
    }
}

// todo(@jroesch): #8800 Follow up with ByteArray RetValue ownership.
// impl From<ByteArray> for RetValue {
//     fn from(val: ByteArray) -> RetValue {
//         match val {
//             ByteArray::Rust(byte_array) | ByteArray::External(byte_array) => {
//                 // TODO(@jroesch): This requires a little more work, going to land narratives
//                 RetValue::Bytes(byte_array)
//             }
//         }
//     }
// }

impl TryFrom<RetValue> for ByteArray {
    type Error = ValueDowncastError;
    fn try_from(val: RetValue) -> Result<ByteArray, Self::Error> {
        match val {
            RetValue::Bytes(array) => Ok(ByteArray::External(array)),
            _ => Err(ValueDowncastError {
                expected_type: "ByteArray",
                actual_type: format!("{:?}", val),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        let v = vec![1u8, 2, 3];
        let barr = ByteArray::from(v.to_vec());
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.to_vec(), vec![1u8, 2, 3]);
        let v = b"hello";
        let barr = ByteArray::from(v.to_vec());
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), &[104u8, 101, 108, 108, 111]);
    }
}
