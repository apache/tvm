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

use std::cmp::{Ordering, PartialEq};
use std::hash::{Hash, Hasher};

use super::Object;

use tvm_macros::Object;

#[repr(C)]
#[derive(Object, Debug)]
#[ref_name = "String"]
#[type_key = "runtime.String"]
#[no_derive]
pub struct StringObj {
    base: Object,
    data: *const u8,
    size: u64,
}

impl From<std::string::String> for String {
    fn from(s: std::string::String) -> Self {
        let size = s.len() as u64;
        let data = Box::into_raw(s.into_boxed_str()).cast();
        let base = Object::base::<StringObj>();
        StringObj { base, data, size }.into()
    }
}

impl From<&'static str> for String {
    fn from(s: &'static str) -> Self {
        let size = s.len() as u64;
        let data = s.as_bytes().as_ptr();
        let base = Object::base::<StringObj>();
        StringObj { base, data, size }.into()
    }
}

impl AsRef<[u8]> for String {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::fmt::Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_string_lossy().fmt(f)
    }
}

impl String {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.size as usize
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len()) }
    }

    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes())
    }

    pub fn to_string_lossy(&self) -> std::borrow::Cow<str> {
        std::string::String::from_utf8_lossy(self.as_bytes())
    }
}

impl<T: AsRef<[u8]>> PartialEq<T> for String {
    fn eq(&self, other: &T) -> bool {
        self.as_bytes() == other.as_ref()
    }
}

impl<T: AsRef<[u8]>> PartialOrd<T> for String {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.as_bytes().partial_cmp(other.as_ref())
    }
}

impl Eq for String {}

impl Ord for String {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl Hash for String {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl std::fmt::Debug for String {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_fmt(format_args!("{:?}", self.to_string_lossy()))
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
        let s = String::from("foo");
        let object_ref = s.upcast();
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
