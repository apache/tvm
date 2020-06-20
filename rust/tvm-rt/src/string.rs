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

use std::ffi::{CString, NulError};
use std::os::raw::c_char;

use super::errors::Error;
use super::{Object, ObjectPtr, ObjectRef};

use tvm_macros::Object;

#[repr(C)]
#[derive(Object)]
#[ref_name = "String"]
#[type_key = "runtime.String"]
pub struct StringObj {
    base: Object,
    data: *const c_char,
    size: u64,
}

impl String {
    pub fn new(string: std::string::String) -> Result<String, Error> {
        let cstring = CString::new(string)?;

        // The string is being corrupted.
        // why is this wrong
        let length = cstring.as_bytes().len();

        let string_obj = StringObj {
            base: Object::base_object::<StringObj>(),
            data: cstring.into_raw(),
            size: length as u64,
        };

        let object_ptr = ObjectPtr::new(string_obj);
        Ok(String(Some(object_ptr)))
    }

    pub fn to_cstring(&self) -> Result<std::ffi::CString, NulError> {
        use std::slice;
        let ptr = self.0.as_ref().unwrap().data;
        let size = self.0.as_ref().unwrap().size;
        unsafe {
            let slice: &[u8] = slice::from_raw_parts(ptr as *const u8, size as usize);
            CString::new(slice)
        }
    }

    pub fn to_string(&self) -> Result<std::string::String, Error> {
        let string = self.to_cstring()?.into_string()?;
        Ok(string)
    }
}

#[cfg(test)]
mod tests {
    use super::String;
    use crate::object::debug_print;
    use crate::IsObjectRef;
    use anyhow::{ensure, Result};

    #[test]
    fn test_string_debug() -> Result<()> {
        let s = String::new("foo".to_string()).unwrap();
        let object_ref = s.to_object_ref();
        println!("about to call");
        let string = debug_print(object_ref)?;
        println!("after call");
        ensure!(
            string.into_string().expect("is cstring").contains("foo"),
            "string content is invalid"
        );
        Ok(())
    }
}
