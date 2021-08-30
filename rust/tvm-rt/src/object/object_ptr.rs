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
use std::fmt;
use std::os::raw::c_char;
use std::ptr::NonNull;
use std::sync::atomic::AtomicI32;

use tvm_macros::Object;
use tvm_sys::ffi::{
    self, TVMObjectFree, TVMObjectRetain, TVMObjectTypeIndex2Key, TVMObjectTypeKey2Index,
};
use tvm_sys::{ArgValue, RetValue};

use crate::errors::Error;

type Deleter = unsafe extern "C" fn(object: *mut Object) -> ();

/// A TVM intrusive smart pointer header, in TVM all FFI compatible types
/// start with an Object as their first field. The base object tracks
/// a type_index which is an index into the runtime type information
/// table, an atomic reference count, and a customized deleter which
/// will be invoked when the reference count is zero.
///
#[derive(Debug, Object)]
#[ref_name = "ObjectRef"]
#[type_key = "runtime.Object"]
#[repr(C)]
pub struct Object {
    /// The index into TVM's runtime type information table.
    pub(self) type_index: u32,
    // TODO(@jroesch): pretty sure Rust and C++ atomics are the same, but not sure.
    // NB: in general we should not touch this in Rust.
    /// The reference count of the smart pointer.
    pub(self) ref_count: AtomicI32,
    /// The deleter function which is used to deallocate the underlying data
    /// when the reference count is zero. This field must always be set for
    /// all objects.
    ///
    /// The common use case is ensuring that the allocator which allocated the
    /// data is also the one that deletes it.
    pub(self) fdeleter: Deleter,
}

/// The default deleter for objects allocated in Rust, we use a bit of
/// trait magic here to get a monomorphized deleter for each object
/// "subtype".
///
/// This function just converts the pointer to the correct type
/// and reconstructs a Box which then is dropped to deallocate
/// the underlying allocation.
unsafe extern "C" fn delete<T: IsObject>(object: *mut Object) {
    let typed_object: *mut T = object as *mut T;
    let boxed: Box<T> = Box::from_raw(typed_object);
    drop(boxed);
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
            // NB(@jroesch): I believe it is sound to use Rust atomics
            // in conjunction with C++ atomics given the memory model
            // is nearly identical.
            //
            // Of course these are famous last words which I may later
            // regret.
            ref_count: AtomicI32::new(0),
            fdeleter: deleter,
        }
    }

    fn get_type_key(&self) -> String {
        let mut cstring: *mut c_char = std::ptr::null_mut();
        unsafe {
            if TVMObjectTypeIndex2Key(self.type_index, &mut cstring as *mut _) != 0 {
                panic!("{}", crate::get_last_error());
            }
            return CString::from_raw(cstring)
                .into_string()
                .expect("type keys should be valid utf-8");
        }
    }

    fn get_type_index<T: IsObject>() -> u32 {
        let type_key = T::TYPE_KEY;
        let cstring = CString::new(type_key).expect("type key must not contain null characters");

        // TODO(@jroesch): look into TVMObjectTypeKey2Index.
        if type_key == "runtime.Object" {
            return 0;
        } else {
            let mut index = 0;
            unsafe {
                if TVMObjectTypeKey2Index(cstring.as_ptr(), &mut index) != 0 {
                    panic!("{}", crate::get_last_error())
                }
            }
            return index;
        }
    }

    pub fn count(&self) -> i32 {
        // need to do atomic read in C++
        // ABI compatible atomics is funky/hard.
        self.ref_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Allocates a base object value for an object subtype of type T.
    /// By using associated constants and generics we can provide a
    /// type indexed abstraction over allocating objects with the
    /// correct index and deleter.
    pub fn base<T: IsObject>() -> Object {
        let index = Object::get_type_index::<T>();
        Object::new(index, delete::<T>)
    }

    /// Increases the object's reference count by one.
    pub(self) fn inc_ref(&self) {
        let raw_ptr = self as *const Object as *mut Object as *mut std::ffi::c_void;
        unsafe {
            assert_eq!(TVMObjectRetain(raw_ptr), 0);
        }
    }

    /// Decreases the object's reference count by one.
    pub(self) fn dec_ref(&self) {
        let raw_ptr = self as *const Object as *mut Object as *mut std::ffi::c_void;
        unsafe {
            assert_eq!(TVMObjectFree(raw_ptr), 0);
        }
    }
}

/// An unsafe trait which should be implemented for an object
/// subtype.
///
/// The trait contains the type key needed to compute the type
/// index, a method for accessing the base object given the
/// subtype, and a typed delete method which is specialized
/// to the subtype.
pub unsafe trait IsObject: AsRef<Object> + std::fmt::Debug {
    const TYPE_KEY: &'static str;
}

/// A smart pointer for types which implement IsObject.
/// This type directly corresponds to TVM's C++ type ObjectPtr<T>.
///
/// See object.h for more details.
#[repr(C)]
pub struct ObjectPtr<T: IsObject> {
    pub ptr: NonNull<T>,
}

impl ObjectPtr<Object> {
    pub fn from_raw(object_ptr: *mut Object) -> Option<ObjectPtr<Object>> {
        let non_null = NonNull::new(object_ptr);
        non_null.map(|ptr| {
            debug_assert!(unsafe { ptr.as_ref().count() } >= 0);
            ObjectPtr { ptr }
        })
    }
}

impl<T: IsObject> Clone for ObjectPtr<T> {
    fn clone(&self) -> Self {
        unsafe { self.ptr.as_ref().as_ref().inc_ref() }
        ObjectPtr { ptr: self.ptr }
    }
}

impl<T: IsObject> Drop for ObjectPtr<T> {
    fn drop(&mut self) {
        unsafe { self.ptr.as_ref().as_ref().dec_ref() }
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
        object.as_ref().inc_ref();
        let object_ptr = Box::new(object);
        let object_ptr = Box::leak(object_ptr);
        let ptr = NonNull::from(object_ptr);
        ObjectPtr { ptr }
    }

    pub fn count(&self) -> i32 {
        // need to do atomic read in C++
        // ABI compatible atomics is funky/hard.
        self.as_ref()
            .ref_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// This method avoid running the destructor on self once it's dropped, so we don't accidentally release the memory
    unsafe fn cast<U: IsObject>(self) -> ObjectPtr<U> {
        let ptr = self.ptr.cast();
        std::mem::forget(self);
        ObjectPtr { ptr }
    }

    pub fn upcast<U>(self) -> ObjectPtr<U>
    where
        U: IsObject,
        T: AsRef<U>,
    {
        unsafe { self.cast() }
    }

    pub fn downcast<U>(self) -> Result<ObjectPtr<U>, Error>
    where
        U: IsObject + AsRef<T>,
    {
        let child_index = Object::get_type_index::<U>();
        let object_index = self.as_ref().type_index;

        let is_derived = if child_index == object_index {
            true
        } else {
            // TODO(@jroesch): write tests
            derived_from(object_index, child_index)
        };

        if is_derived {
            Ok(unsafe { self.cast() })
        } else {
            let type_key = self.as_ref().get_type_key();
            Err(Error::downcast(type_key.into(), U::TYPE_KEY))
        }
    }

    pub unsafe fn into_raw(self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub unsafe fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T: IsObject> std::ops::Deref for ObjectPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: IsObject> fmt::Debug for ObjectPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::ops::Deref;
        write!(f, "{:?}", self.deref())
    }
}

impl<'a, T: IsObject> From<ObjectPtr<T>> for RetValue {
    fn from(object_ptr: ObjectPtr<T>) -> RetValue {
        let raw_object_ptr = ObjectPtr::leak(object_ptr) as *mut T as *mut std::ffi::c_void;
        assert!(!raw_object_ptr.is_null());
        RetValue::ObjectHandle(raw_object_ptr)
    }
}

impl<'a, T: IsObject> TryFrom<RetValue> for ObjectPtr<T> {
    type Error = Error;

    fn try_from(ret_value: RetValue) -> Result<ObjectPtr<T>, Self::Error> {
        use crate::ffi::DLTensor;
        use crate::ndarray::NDArrayContainer;

        match ret_value {
            RetValue::ObjectHandle(handle) | RetValue::ModuleHandle(handle) => {
                let optr = ObjectPtr::from_raw(handle as *mut Object).ok_or(Error::Null)?;
                debug_assert!(optr.count() >= 1);
                optr.downcast()
            }
            RetValue::NDArrayHandle(handle) => {
                let optr: ObjectPtr<NDArrayContainer> =
                    NDArrayContainer::from_raw(handle as *mut DLTensor).ok_or(Error::Null)?;
                debug_assert!(optr.count() >= 1);
                optr.upcast::<Object>().downcast()
            }
            _ => Err(Error::downcast(format!("{:?}", ret_value), T::TYPE_KEY)),
        }
    }
}

impl<'a, T: IsObject> From<&'a ObjectPtr<T>> for ArgValue<'a> {
    fn from(object_ptr: &'a ObjectPtr<T>) -> ArgValue<'a> {
        debug_assert!(object_ptr.count() >= 1);
        let object_ptr = object_ptr.clone().upcast::<Object>();
        match T::TYPE_KEY {
            "runtime.NDArray" => {
                use crate::ndarray::NDArrayContainer;
                let dcast_ptr = object_ptr.downcast().unwrap();
                let raw_ptr = NDArrayContainer::as_mut_ptr(&dcast_ptr) as *mut std::ffi::c_void;
                assert!(!raw_ptr.is_null());
                ArgValue::NDArrayHandle(raw_ptr)
            }
            "runtime.Module" => {
                let raw_ptr = unsafe { object_ptr.as_ptr() } as *mut std::ffi::c_void;
                assert!(!raw_ptr.is_null());
                ArgValue::ModuleHandle(raw_ptr)
            }
            _ => {
                let raw_ptr = unsafe { object_ptr.as_ptr() } as *mut std::ffi::c_void;
                assert!(!raw_ptr.is_null());
                ArgValue::ObjectHandle(raw_ptr)
            }
        }
    }
}

impl<'a, T: IsObject> TryFrom<ArgValue<'a>> for ObjectPtr<T> {
    type Error = Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<ObjectPtr<T>, Self::Error> {
        use crate::ffi::DLTensor;
        use crate::ndarray::NDArrayContainer;

        match arg_value {
            ArgValue::ObjectHandle(handle) | ArgValue::ModuleHandle(handle) => {
                let optr = ObjectPtr::from_raw(handle as *mut Object).ok_or(Error::Null)?;
                optr.inc_ref();
                // We are building an owned, ref-counted view into the underlying ArgValue, in order to be safe we must
                // bump the reference count by one.
                assert!(optr.count() >= 1);
                optr.downcast()
            }
            ArgValue::NDArrayHandle(handle) => {
                let optr =
                    NDArrayContainer::from_raw(handle as *mut DLTensor).ok_or(Error::Null)?;
                // We are building an owned, ref-counted view into the underlying ArgValue, in order to be safe we must
                // bump the reference count by one.
                assert!(optr.count() >= 1);
                // TODO(@jroesch): figure out if there is a more optimal way to do this
                let object = optr.upcast::<Object>();
                object.inc_ref();
                object.downcast()
            }
            _ => Err(Error::downcast(format!("{:?}", arg_value), "ObjectHandle")),
        }
    }
}

impl<T: IsObject> std::hash::Hash for ObjectPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_i64(
            super::structural_hash(ObjectRef(Some(self.clone().upcast())), false).unwrap(),
        )
    }
}

impl<T: IsObject> PartialEq for ObjectPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        let lhs = ObjectRef(Some(self.clone().upcast()));
        let rhs = ObjectRef(Some(other.clone().upcast()));
        super::structural_equal(lhs, rhs, false, false).unwrap()
    }
}

impl<T: IsObject> Eq for ObjectPtr<T> {}

#[cfg(test)]
mod tests {
    use super::{Object, ObjectPtr};
    use anyhow::{ensure, Result};
    use std::convert::TryInto;
    use tvm_sys::{ArgValue, RetValue};

    #[test]
    fn test_new_object() -> anyhow::Result<()> {
        let object = Object::base::<Object>();
        let ptr = ObjectPtr::new(object);
        assert_eq!(ptr.count(), 1);
        Ok(())
    }

    #[test]
    fn test_leak() -> anyhow::Result<()> {
        let ptr = ObjectPtr::new(Object::base::<Object>());
        assert_eq!(ptr.count(), 1);
        let object = ObjectPtr::leak(ptr);
        assert_eq!(object.count(), 1);
        Ok(())
    }

    #[test]
    fn test_clone() -> anyhow::Result<()> {
        let ptr = ObjectPtr::new(Object::base::<Object>());
        assert_eq!(ptr.count(), 1);
        let ptr2 = ptr.clone();
        assert_eq!(ptr2.count(), 2);
        drop(ptr);
        assert_eq!(ptr2.count(), 1);
        Ok(())
    }

    #[test]
    fn roundtrip_retvalue() -> Result<()> {
        let ptr = ObjectPtr::new(Object::base::<Object>());
        assert_eq!(ptr.count(), 1);
        let ret_value: RetValue = ptr.clone().into();
        let ptr2: ObjectPtr<Object> = ret_value.try_into()?;
        assert_eq!(ptr.count(), ptr2.count());
        assert_eq!(ptr.count(), 2);
        ensure!(
            ptr.type_index == ptr2.type_index,
            "type indices do not match"
        );
        ensure!(
            ptr.fdeleter == ptr2.fdeleter,
            "objects have different deleters"
        );
        // After dropping the second pointer we should only see only refcount.
        drop(ptr2);
        assert_eq!(ptr.count(), 1);
        Ok(())
    }

    #[test]
    fn roundtrip_argvalue() -> Result<()> {
        let ptr = ObjectPtr::new(Object::base::<Object>());
        assert_eq!(ptr.count(), 1);
        let ptr_clone = ptr.clone();
        assert_eq!(ptr.count(), 2);
        let arg_value: ArgValue = (&ptr_clone).into();
        assert_eq!(ptr.count(), 2);
        let ptr2: ObjectPtr<Object> = arg_value.try_into()?;
        assert_eq!(ptr2.count(), 3);
        assert_eq!(ptr.count(), ptr2.count());
        drop(ptr_clone);
        assert_eq!(ptr.count(), 2);
        ensure!(
            ptr.type_index == ptr2.type_index,
            "type indices do not match"
        );
        ensure!(
            ptr.fdeleter == ptr2.fdeleter,
            "objects have different deleters"
        );
        // After dropping the second pointer we should only see only refcount.
        drop(ptr2);
        assert_eq!(ptr.count(), 1);
        Ok(())
    }

    fn test_fn_raw<'a>(
        mut args: crate::to_function::ArgList<'a>,
    ) -> crate::function::Result<RetValue> {
        let v: ArgValue = args.remove(0);
        let v2: ArgValue = args.remove(0);
        // assert_eq!(o.count(), 2);
        let o: ObjectPtr<Object> = v.try_into().unwrap();
        assert_eq!(o.count(), 2);
        let o2: ObjectPtr<Object> = v2.try_into().unwrap();
        assert_eq!(o2.count(), 3);
        drop(o2);
        assert_eq!(o.count(), 2);
        Ok(o.into())
    }

    #[test]
    fn test_ref_count_raw_fn() {
        use super::*;
        use crate::function::{register_untyped, Function};
        let ptr = ObjectPtr::new(Object::base::<Object>());
        // Call the function without the wrapping for TVM.
        assert_eq!(ptr.count(), 1);
        let same = test_fn_raw(vec![(&ptr).into(), (&ptr).into()]).unwrap();
        let output: ObjectPtr<Object> = same.try_into().unwrap();
        assert_eq!(output.count(), 2);
        drop(output);
        assert_eq!(ptr.count(), 1);

        register_untyped(test_fn_raw, "test_fn_raw", true).unwrap();
        let raw_func = Function::get("test_fn_raw").unwrap();
        let output = raw_func.invoke(vec![(&ptr).into(), (&ptr).into()]).unwrap();
        let output: ObjectPtr<Object> = output.try_into().unwrap();
        assert_eq!(output.count(), 2);
        drop(output);
        assert_eq!(ptr.count(), 1);
    }

    fn test_fn_typed(o: ObjectPtr<Object>, o2: ObjectPtr<Object>) -> ObjectPtr<Object> {
        assert_eq!(o.count(), 3);
        assert_eq!(o2.count(), 3);
        drop(o2);
        assert_eq!(o.count(), 2);
        return o;
    }

    #[test]
    fn test_ref_count_typed() {
        use super::*;
        use crate::function::{register, Function};
        let ptr = ObjectPtr::new(Object::base::<Object>());
        // Call the function without the wrapping for TVM.
        assert_eq!(ptr.count(), 1);
        let output = test_fn_typed(ptr.clone(), ptr.clone());
        assert_eq!(output.count(), 2);
        drop(output);
        assert_eq!(ptr.count(), 1);

        register(test_fn_typed, "test_fn_typed").unwrap();
        let typed_func = Function::get("test_fn_typed").unwrap();
        let output = typed_func
            .invoke(vec![(&ptr).into(), (&ptr).into()])
            .unwrap();
        let output: ObjectPtr<Object> = output.try_into().unwrap();
        assert_eq!(output.count(), 2);
        drop(output);
        assert_eq!(ptr.count(), 1);
    }
}
