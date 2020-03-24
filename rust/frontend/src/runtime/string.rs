use std::ffi::{CString, NulError};
use std::os::raw::{c_char};

use super::{Object, IsObject, ObjectPtr};

#[repr(C)]
pub struct StringObj {
    base: Object,
    data: *const c_char,
    size: u64,
}


unsafe impl IsObject for StringObj {
    const TYPE_KEY: &'static str = "runtime.String";

    fn as_object<'s>(&'s self) -> &'s Object {
        &self.base
    }
}

pub struct String(ObjectPtr<StringObj>);

impl String {
    pub fn new(string: std::string::String) -> Result<String, NulError> {
        let cstring = CString::new(string)?;
        let length = cstring.as_bytes().len();

        let string_obj = StringObj {
            base: Object::base_object::<StringObj>(),
            data: cstring.into_raw(),
            size: length as u64,
        };

        let object_ptr = ObjectPtr::new(string_obj);
        Ok(String(object_ptr))
    }
}
