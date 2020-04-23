use std::convert::TryFrom;
use std::ffi::CString;
use std::ptr::NonNull;
use tvm_sys::ffi::{self, /* TVMObjectFree, */ TVMObjectRetain, TVMObjectTypeKey2Index};
use tvm_sys::{TVMArgValue, TVMRetValue};
use anyhow::Context;

type Deleter<T> = unsafe extern "C" fn(object: *mut T) -> ();

#[derive(Debug)]
#[repr(C)]
pub struct Object {
    pub type_index: u32,
    pub ref_count: i32,
    pub fdeleter: Deleter<Object>,
}

unsafe extern "C" fn delete<T: IsObject>(object: *mut Object) {
    let typed_object: *mut T = std::mem::transmute(object);
    T::typed_delete(typed_object);
}

fn derived_from(child_type_index: u32, parent_type_index: u32) -> bool {
    let mut is_derived = 0;
    crate::check_call!(ffi::TVMObjectDerivedFrom(
        child_type_index,
        parent_type_index,
        &mut is_derived
    ));
    if is_derived == 0 {
        false
    } else {
        true
    }
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

    fn get_type_index<T: IsObject>() -> u32 {
        let type_key = T::TYPE_KEY;
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
        let index = Object::get_type_index::<T>();
        Object::new(index, delete::<T>)
    }
}

pub unsafe trait IsObject {
    const TYPE_KEY: &'static str;

    fn as_object<'s>(&'s self) -> &'s Object;

    unsafe extern "C" fn typed_delete(_object: *mut Self) {
        // let object = Box::from_raw(object);
        // drop(object)
    }
}

unsafe impl IsObject for Object {
    const TYPE_KEY: &'static str = "Object";

    fn as_object<'s>(&'s self) -> &'s Object {
        self
    }
}

#[repr(C)]
pub struct ObjectPtr<T> {
    pub ptr: NonNull<T>,
}

impl ObjectPtr<Object> {
    fn from_raw(object_ptr: *mut Object) -> Option<ObjectPtr<Object>> {
        println!("{:?}", object_ptr);
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

// impl<T> Drop for ObjectPtr<T> {
//     fn drop(&mut self) {
//         unsafe {
//             let raw_ptr = std::mem::transmute(self.ptr);
//             assert_eq!(TVMObjectFree(raw_ptr), 0)
//         }
//     }
// }

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

    pub fn upcast(&self) -> ObjectPtr<Object> {
        ObjectPtr {
            ptr: self.ptr.cast(),
        }
    }

    pub fn downcast<U: IsObject>(&self) -> anyhow::Result<ObjectPtr<U>> {
        let child_index = Object::get_type_index::<U>();
        let object_index = self.as_object().type_index;

        let is_derived = if child_index == object_index {
            true
        } else {
            // TODO(@jroesch): write tests
            derived_from(object_index, child_index)
        };

        if is_derived {
            Ok(ObjectPtr {
                ptr: self.ptr.cast(),
            })
        } else {
            Err(anyhow::anyhow!("failed to downcast to object subtype"))
        }
    }
}

impl<T> std::ops::Deref for ObjectPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T: IsObject> From<ObjectPtr<T>> for TVMRetValue {
    fn from(object_ptr: ObjectPtr<T>) -> TVMRetValue {
        let raw_object_ptr = object_ptr.ptr.as_ptr();
        // Should be able to hide this unsafety in raw bindings.
        let void_ptr = unsafe { std::mem::transmute(raw_object_ptr) };
        TVMRetValue::ObjectHandle(void_ptr)
    }
}

impl<'a, T: IsObject> TryFrom<TVMRetValue> for ObjectPtr<T> {
    type Error = anyhow::Error;

    fn try_from(ret_value: TVMRetValue) -> Result<ObjectPtr<T>, Self::Error> {
        match ret_value {
            TVMRetValue::ObjectHandle(handle) => {
                let handle: *mut Object = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle)
                       .context("unable to convert nullptr")?;
                    optr.downcast()
            }
            _ => Err(anyhow::anyhow!("unable to convert the result to an Object")),
        }
    }
}

impl<'a, T: IsObject> From<ObjectPtr<T>> for TVMArgValue<'a> {
    fn from(object_ptr: ObjectPtr<T>) -> TVMArgValue<'a> {
        let raw_object_ptr = object_ptr.ptr.as_ptr();
        // Should be able to hide this unsafety in raw bindings.
        let void_ptr = unsafe { std::mem::transmute(raw_object_ptr) };
        TVMArgValue::ObjectHandle(void_ptr)
    }
}

impl<'a, T: IsObject> TryFrom<TVMArgValue<'a>> for ObjectPtr<T> {
    type Error = anyhow::Error;
    fn try_from(arg_value: TVMArgValue<'a>) -> Result<ObjectPtr<T>, Self::Error> {
        match arg_value {
            TVMArgValue::ObjectHandle(handle) => {
                let handle = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle)
                    .context("unable to convert nullptr")?;
                optr.downcast()
            }
            _ => Err(anyhow::anyhow!("unable to convert the result to an Object")),
        }
    }
}

impl<'a, T: IsObject> TryFrom<&TVMArgValue<'a>> for ObjectPtr<T> {
    type Error = anyhow::Error;
    fn try_from(arg_value: &TVMArgValue<'a>) -> Result<ObjectPtr<T>, Self::Error> {
        match arg_value {
            TVMArgValue::ObjectHandle(handle) => {
                let handle = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle)
                    .context("unable to convert nullptr")?;
                optr.downcast()
            }
            _ => Err(anyhow::anyhow!("unable to convert the result to an Object")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Object, ObjectPtr};
    use tvm_sys::{TVMArgValue, TVMRetValue};
    use anyhow::{Result, ensure};
    use std::convert::TryInto;

    #[test]
    fn test_new_object() -> anyhow::Result<()> {
        let object = Object::base_object::<Object>();
        let ptr = ObjectPtr::new(object);
        assert_eq!(ptr.count(), 1);
        Ok(())
    }

    #[test]
    fn roundtrip_retvalue() -> Result<()> {
        let ptr = ObjectPtr::new(Object::base_object::<Object>());
        let ret_value: TVMRetValue = ptr.clone().into();
        let ptr2: ObjectPtr<Object> = ret_value.try_into()?;
        ensure!(ptr.type_index == ptr2.type_index, "type indices do not match");
        ensure!(ptr.fdeleter == ptr2.fdeleter, "objects have different deleters");
        Ok(())
    }



}
