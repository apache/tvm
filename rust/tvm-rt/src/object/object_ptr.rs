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
use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::atomic::AtomicI32;

use tvm_sys::ffi::{self, TVMObjectFree, TVMObjectRetain, TVMObjectTypeKey2Index};
use tvm_sys::{ArgValue, RetValue};

use crate::errors::Error;

type Deleter = unsafe extern "C" fn(object: *mut Object) -> ();

#[derive(Debug)]
#[repr(C)]
pub struct Object {
    pub type_index: u32,
    // TODO(@jroesch): pretty sure Rust and C++ atomics are the same, but not sure.
    // NB: in general we should not touch this in Rust.
    pub(self) ref_count: AtomicI32,
    pub fdeleter: Deleter,
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
    fn new(type_index: u32, deleter: Deleter) -> Object {
        Object {
            type_index,
            // Note: do not touch this field directly again, this is
            // a critical section, we write a 1 to the atomic which will now
            // be managed by the C++ atomics.
            // In the future we should probably use C-atomcis.
            ref_count: AtomicI32::new(0),
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

    pub(self) fn inc_ref(&self) {
        unsafe {
            let raw_ptr = std::mem::transmute(self);
            assert_eq!(TVMObjectRetain(raw_ptr), 0);
        }
    }

    pub(self) fn dec_ref(&self) {
        unsafe {
            let raw_ptr = std::mem::transmute(self);
            assert_eq!(TVMObjectFree(raw_ptr), 0);
        }
    }
}

pub unsafe trait IsObject {
    const TYPE_KEY: &'static str;

    fn as_object<'s>(&'s self) -> &'s Object;

    unsafe extern "C" fn typed_delete(object: *mut Self) {
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

#[repr(C)]
pub struct ObjectPtr<T: IsObject> {
    pub ptr: NonNull<T>,
}

fn inc_ref<T: IsObject>(ptr: NonNull<T>) {
    unsafe { ptr.as_ref().as_object().inc_ref() }
}

fn dec_ref<T: IsObject>(ptr: NonNull<T>) {
    unsafe { ptr.as_ref().as_object().dec_ref() }
}

impl ObjectPtr<Object> {
    fn from_raw(object_ptr: *mut Object) -> Option<ObjectPtr<Object>> {
        let non_null = NonNull::new(object_ptr);
        non_null.map(|ptr| ObjectPtr { ptr })
    }
}

impl<T: IsObject> Clone for ObjectPtr<T> {
    fn clone(&self) -> Self {
        inc_ref(self.ptr);
        ObjectPtr { ptr: self.ptr }
    }
}

impl<T: IsObject> Drop for ObjectPtr<T> {
    fn drop(&mut self) {
        dec_ref(self.ptr);
    }
}

impl<T: IsObject> ObjectPtr<T> {
    pub fn leak<'a>(object_ptr: ObjectPtr<T>) -> &'a mut T
    where
        T: 'a,
    {
        unsafe { &mut *std::mem::ManuallyDrop::new(object_ptr).ptr.as_ptr() }
    }

    pub fn new(object: T) -> ObjectPtr<T> {
        let object_ptr = Box::new(object);
        let object_ptr = Box::leak(object_ptr);
        let ptr = NonNull::from(object_ptr);
        inc_ref(ptr);
        ObjectPtr { ptr }
    }

    pub fn count(&self) -> i32 {
        // need to do atomic read in C++
        // ABI compatible atomics is funky/hard.
        self.as_object()
            .ref_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    fn as_object<'s>(&'s self) -> &'s Object {
        unsafe { self.ptr.as_ref().as_object() }
    }

    pub fn upcast(&self) -> ObjectPtr<Object> {
        ObjectPtr {
            ptr: self.ptr.cast(),
        }
    }

    pub fn downcast<U: IsObject>(&self) -> Result<ObjectPtr<U>, Error> {
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
            Err(Error::downcast("TODOget_type_key".into(), U::TYPE_KEY))
        }
    }
}

impl<T: IsObject> std::ops::Deref for ObjectPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T: IsObject> From<ObjectPtr<T>> for RetValue {
    fn from(object_ptr: ObjectPtr<T>) -> RetValue {
        let raw_object_ptr = ObjectPtr::leak(object_ptr);
        let void_ptr = unsafe { std::mem::transmute(raw_object_ptr) };
        RetValue::ObjectHandle(void_ptr)
    }
}

impl<'a, T: IsObject> TryFrom<RetValue> for ObjectPtr<T> {
    type Error = Error;

    fn try_from(ret_value: RetValue) -> Result<ObjectPtr<T>, Self::Error> {
        match ret_value {
            RetValue::ObjectHandle(handle) => {
                let handle: *mut Object = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle).ok_or(Error::Null)?;
                optr.downcast()
            }
            _ => Err(Error::downcast(format!("{:?}", ret_value), "ObjectHandle")),
        }
    }
}

impl<'a, T: IsObject> From<ObjectPtr<T>> for ArgValue<'a> {
    fn from(object_ptr: ObjectPtr<T>) -> ArgValue<'a> {
        let raw_object_ptr = ObjectPtr::leak(object_ptr);
        let void_ptr = unsafe { std::mem::transmute(raw_object_ptr) };
        ArgValue::ObjectHandle(void_ptr)
    }
}

impl<'a, T: IsObject> TryFrom<ArgValue<'a>> for ObjectPtr<T> {
    type Error = Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<ObjectPtr<T>, Self::Error> {
        match arg_value {
            ArgValue::ObjectHandle(handle) => {
                let handle = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle).ok_or(Error::Null)?;
                optr.downcast()
            }
            _ => Err(Error::downcast(format!("{:?}", arg_value), "ObjectHandle")),
        }
    }
}

impl<'a, T: IsObject> TryFrom<&ArgValue<'a>> for ObjectPtr<T> {
    type Error = Error;

    fn try_from(arg_value: &ArgValue<'a>) -> Result<ObjectPtr<T>, Self::Error> {
        match arg_value {
            ArgValue::ObjectHandle(handle) => {
                let handle = unsafe { std::mem::transmute(handle) };
                let optr = ObjectPtr::from_raw(handle).ok_or(Error::Null)?;
                optr.downcast()
            }
            _ => Err(Error::downcast(format!("{:?}", arg_value), "ObjectHandle")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Object, ObjectPtr};
    use anyhow::{ensure, Result};
    use std::convert::TryInto;
    use tvm_sys::{ArgValue, RetValue};

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
        let ret_value: RetValue = ptr.clone().into();
        let ptr2: ObjectPtr<Object> = ret_value.try_into()?;
        ensure!(
            ptr.type_index == ptr2.type_index,
            "type indices do not match"
        );
        ensure!(
            ptr.fdeleter == ptr2.fdeleter,
            "objects have different deleters"
        );
        Ok(())
    }

    #[test]
    fn roundtrip_argvalue() -> Result<()> {
        let ptr = ObjectPtr::new(Object::base_object::<Object>());
        let arg_value: ArgValue = ptr.clone().into();
        let ptr2: ObjectPtr<Object> = arg_value.try_into()?;
        ensure!(
            ptr.type_index == ptr2.type_index,
            "type indices do not match"
        );
        ensure!(
            ptr.fdeleter == ptr2.fdeleter,
            "objects have different deleters"
        );
        Ok(())
    }

    fn test_fn(o: ObjectPtr<Object>) -> ObjectPtr<Object> {
        assert_eq!(o.count(), 2);
        return o;
    }

    #[test]
    fn test_ref_count_boundary() {
        use super::*;
        use crate::function::{register, Function, Result};
        let ptr = ObjectPtr::new(Object::base_object::<Object>());
        let stay = ptr.clone();
        assert_eq!(ptr.count(), 2);
        register(test_fn, "my_func").unwrap();
        let func = Function::get("my_func").unwrap();
        let func = func.to_boxed_fn::<dyn Fn(ObjectPtr<Object>) -> Result<ObjectPtr<Object>>>();
        func(ptr).unwrap();
        assert_eq!(stay.count(), 1);
    }
}
