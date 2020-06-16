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
use std::convert::TryInto;
use std::ffi::CString;

use crate::errors::Error;
use crate::external;

use tvm_sys::{ArgValue, RetValue};

mod object_ptr;

pub use object_ptr::{IsObject, Object, ObjectPtr};

#[derive(Clone)]
pub struct ObjectRef(pub Option<ObjectPtr<Object>>);

impl ObjectRef {
    pub fn null() -> ObjectRef {
        ObjectRef(None)
    }
}

pub trait IsObjectRef: Sized {
    type Object: IsObject;
    fn as_object_ptr(&self) -> Option<&ObjectPtr<Self::Object>>;
    fn from_object_ptr(object_ptr: Option<ObjectPtr<Self::Object>>) -> Self;

    fn to_object_ref(&self) -> ObjectRef {
        let object_ptr = self.as_object_ptr().cloned();
        ObjectRef(object_ptr.map(|ptr| ptr.upcast()))
    }

    fn downcast<U: IsObjectRef>(&self) -> Result<U, Error> {
        let ptr = self.as_object_ptr().map(|ptr| ptr.downcast::<U::Object>());
        let ptr = ptr.transpose()?;
        Ok(U::from_object_ptr(ptr))
    }
}

impl IsObjectRef for ObjectRef {
    type Object = Object;

    fn as_object_ptr(&self) -> Option<&ObjectPtr<Self::Object>> {
        self.0.as_ref()
    }

    fn from_object_ptr(object_ptr: Option<ObjectPtr<Self::Object>>) -> Self {
        ObjectRef(object_ptr)
    }
}

impl TryFrom<RetValue> for ObjectRef {
    type Error = Error;

    fn try_from(ret_val: RetValue) -> Result<ObjectRef, Self::Error> {
        let optr = ret_val.try_into()?;
        Ok(ObjectRef(Some(optr)))
    }
}

impl From<ObjectRef> for RetValue {
    fn from(object_ref: ObjectRef) -> RetValue {
        use std::ffi::c_void;
        let object_ptr = object_ref.0;
        match object_ptr {
            None => RetValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void),
            Some(value) => value.clone().into(),
        }
    }
}

impl<'a> std::convert::TryFrom<ArgValue<'a>> for ObjectRef {
    type Error = Error;

    fn try_from(arg_value: ArgValue<'a>) -> Result<ObjectRef, Self::Error> {
        let optr: ObjectPtr<Object> = arg_value.try_into()?;
        debug_assert!(optr.count() >= 1);
        Ok(ObjectRef(Some(optr)))
    }
}

impl<'a> From<ObjectRef> for ArgValue<'a> {
    fn from(object_ref: ObjectRef) -> ArgValue<'a> {
        use std::ffi::c_void;
        let object_ptr = object_ref.0;
        match object_ptr {
            None => ArgValue::ObjectHandle(std::ptr::null::<c_void>() as *mut c_void),
            Some(value) => value.into(),
        }
    }
}

external! {
    #[name("ir.DebugPrint")]
    fn debug_print(object: ObjectRef) -> CString;
}

// external! {
//     #[name("ir.TextPrinter")]
//     fn as_text(object: ObjectRef) -> CString;
// }
