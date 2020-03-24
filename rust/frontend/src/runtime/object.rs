use std::ffi::{CString};
use crate::Function;
use tvm_common::{TVMRetValue, TVMArgValue};
use tvm_common::ffi::{TVMObjectRetain, TVMObjectFree, TVMObjectTypeKey2Index};
use std::ptr::NonNull;
use std::convert::TryFrom;

type Deleter<T> = unsafe extern fn(object: *mut T) -> ();

#[derive(Debug)]
#[repr(C)]
pub struct Object {
    pub type_index: u32,
    pub ref_count: i32,
    pub fdeleter: Deleter<Object>,
}

unsafe extern fn delete<T: IsObject>(object: *mut Object) {
    let typed_object: *mut T =
        std::mem::transmute(object);
    T::typed_delete(typed_object);
}

impl Object {
    fn new(type_index: u32, deleter: Deleter<Object>) -> Object {
        Object {
            type_index,
            // Note: do not touch this field directly again, this is
            // a critical section, we write a 1 to the atomic which will now
            // be managed by the C++ atomics.
            // In the future we should probably use C-atomcis.
            ref_count: 1,
            fdeleter: deleter,
        }
    }

    fn get_type_index(type_key: &'static str) -> u32 {
        let cstring = CString::new(type_key).expect("type key must not contain null characters");
        if type_key == "Object" {
            return 0;
        } else {
            let mut index = 0;
            unsafe {
                let index_ptr = std::mem::transmute(&mut index);
                if TVMObjectTypeKey2Index(cstring.as_ptr(), index_ptr) != 0 {
                    panic!(crate::get_last_error())
                }
            }
            return index;
        }
    }

    pub fn base_object<T: IsObject>() -> Object {
        let index = Object::get_type_index(T::TYPE_KEY);
        Object::new(index, delete::<T>)
    }
}

pub unsafe trait IsObject {
    const TYPE_KEY: &'static str;

    fn as_object<'s>(&'s self) -> &'s Object;

    unsafe extern fn typed_delete(object: *mut Self) {
        let object = Box::from_raw(object);
        drop(object)
    }
}

unsafe impl IsObject for Object {
    const TYPE_KEY: &'static str = "Object";

    fn as_object<'s>(&'s self) -> &'s Object {
        self
    }
}

// unsafe impl<T: IsObject> IsObject for ObjectPtr<T> {
//     fn as_object<'s>(&'s self) -> &'s Object {
//         unsafe { self.ptr.as_ref().as_object() }
//     }
// }

#[repr(C)]
pub struct ObjectPtr<T> {
    ptr: NonNull<T>,
}

impl ObjectPtr<Object> {
    fn from_raw(object_ptr: *mut Object) -> Option<ObjectPtr<Object>> {
        let non_null = NonNull::new(object_ptr);
        non_null.map(|ptr| ObjectPtr { ptr })
    }
}

impl<T> Clone for ObjectPtr<T> {
    fn clone(&self) -> Self {
        unsafe {
            let raw_ptr = std::mem::transmute(self.ptr);
            assert_eq!(TVMObjectRetain(raw_ptr), 0);
            ObjectPtr { ptr: self.ptr }
        }
    }
}

impl<T> Drop for ObjectPtr<T> {
    fn drop(&mut self) {
        unsafe {
            let raw_ptr = std::mem::transmute(self.ptr);
            assert_eq!(TVMObjectFree(raw_ptr), 0)
        }
    }
}

impl<T: IsObject> ObjectPtr<T> {
    pub fn new(object: T) -> ObjectPtr<T> {
        let object_ptr = Box::new(object);
        let ptr = NonNull::from(Box::leak(object_ptr));
        ObjectPtr { ptr }
    }

    pub fn count(&self) -> i32 {
        // need to do atomic read in C++
        // ABI compatible atomics is funky/hard.
        self.as_object().ref_count
    }

    fn as_object<'s>(&'s self) -> &'s Object {
        unsafe { self.ptr.as_ref().as_object() }
    }
}

pub struct ObjectRef(pub Option<ObjectPtr<Object>>);


impl TryFrom<TVMRetValue> for ObjectRef {
    type Error = ();

    fn try_from(ret_val: TVMRetValue) -> Result<ObjectRef, Self::Error> {
        match ret_val {
            TVMRetValue::ObjectHandle(handle) =>
            // I think we can type the lower-level bindings even further.
            unsafe { let handle = std::mem::transmute(handle);
                Ok(ObjectRef(ObjectPtr::from_raw(handle)))
            },
            _ => Err(())
        }
    }
}

impl<'a> From<&ObjectRef> for TVMArgValue<'a> {
    fn from(object_ref: &ObjectRef) -> TVMArgValue<'a> {
        let object_ptr = &object_ref.0;
        let raw_object_ptr = object_ptr.as_ref().map(|p| p.ptr.as_ptr()).unwrap_or(std::ptr::null_mut());
        // Should be able to hide this unsafety in raw bindings.
        let void_ptr = unsafe { std::mem::transmute(raw_object_ptr) };
        TVMArgValue::ObjectHandle(void_ptr)
    }
}


lazy_static! {
    static ref _DEBUG_PRINT: &'static Function = {
        Function::get("ir.DebugPrinter").expect("ir.DebugPrinter is unregistered")
    };
}

pub fn debug_print(object: &ObjectRef) -> CString {
        let dp: &Function = &_DEBUG_PRINT;
        let ret = crate::call_packed!(dp, object).expect("always returns strings");
        match ret {
            TVMRetValue::Str(cstring) => cstring.into(),
            x => panic!("{:?}", x),
        }
}
