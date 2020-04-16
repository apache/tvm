use std::ffi::{CString, NulError};
use std::os::raw::{c_char};

use super::{Object, IsObject, ObjectRef, ObjectPtr, debug_print};

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

pub struct String(Option<ObjectPtr<StringObj>>);

impl String {
    fn upcast(&self) -> ObjectRef {
        ObjectRef(self.0.as_ref().map(|o| o.upcast()))
    }
}

impl String {
    pub fn new(string: std::string::String) -> Result<String, NulError> {
        let cstring = CString::new(string)?;
        println!("{:?}", cstring);
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
}

#[cfg(test)]
mod tests {
    use super::String;
    use super::{Object, IsObject, ObjectRef, ObjectPtr, debug_print};

    #[test]
    fn test_string_debug() {
        let s = String::new("foo".to_string()).unwrap();
        assert!(debug_print(&s.upcast()).into_string().expect("is cstring").contains("foo"))
    }
}
