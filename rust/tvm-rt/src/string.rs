use std::ffi::{CString, NulError};
use std::os::raw::c_char;

use super::{Object, ObjectPtr, ObjectRef};
use super::errors::Error;

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
    pub fn new(string: std::string::String) -> Result<String, NulError> {
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
    use crate::ToObjectRef;
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
