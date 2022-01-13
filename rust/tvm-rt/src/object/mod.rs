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

use crate::errors::Error;
use crate::external;

use tvm_sys::{ArgValue, RetValue};

mod object_ptr;

pub use object_ptr::{IsObject, Object, ObjectPtr, ObjectRef};

pub trait AsArgValue<'a> {
    fn as_arg_value(&'a self) -> ArgValue<'a>;
}

impl<'a, T: 'static> AsArgValue<'a> for T
where
    &'a T: Into<ArgValue<'a>>,
{
    fn as_arg_value(&'a self) -> ArgValue<'a> {
        self.into()
    }
}

// TODO we would prefer to blanket impl From/TryFrom ArgValue/RetValue, but we
// can't because of coherence rules. Instead, we generate them in the macro, and
// add what we can (including Into instead of From) as subtraits.
// We also add named conversions for clarity
pub trait IsObjectRef:
    Sized
    + Clone
    + Into<RetValue>
    + for<'a> AsArgValue<'a>
    + TryFrom<RetValue, Error = Error>
    + for<'a> TryFrom<ArgValue<'a>, Error = Error>
    + std::fmt::Debug
{
    type Object: IsObject;
    fn as_ptr(&self) -> Option<&ObjectPtr<Self::Object>>;
    fn into_ptr(self) -> Option<ObjectPtr<Self::Object>>;
    fn from_ptr(object_ptr: Option<ObjectPtr<Self::Object>>) -> Self;

    fn null() -> Self {
        Self::from_ptr(None)
    }

    fn into_arg_value<'a>(&'a self) -> ArgValue<'a> {
        self.as_arg_value()
    }

    fn from_arg_value<'a>(arg_value: ArgValue<'a>) -> Result<Self, Error> {
        Self::try_from(arg_value)
    }

    fn into_ret_value<'a>(self) -> RetValue {
        self.into()
    }

    fn from_ret_value<'a>(ret_value: RetValue) -> Result<Self, Error> {
        Self::try_from(ret_value)
    }

    fn upcast<U>(self) -> U
    where
        U: IsObjectRef,
        Self::Object: AsRef<U::Object>,
    {
        let ptr = self.into_ptr().map(ObjectPtr::upcast);
        U::from_ptr(ptr)
    }

    fn downcast<U>(self) -> Result<U, Error>
    where
        U: IsObjectRef,
        U::Object: AsRef<Self::Object>,
    {
        let ptr = self.into_ptr().map(ObjectPtr::downcast);
        let ptr = ptr.transpose()?;
        Ok(U::from_ptr(ptr))
    }
}

external! {
    #[name("ir.DebugPrint")]
    pub fn debug_print(object: ObjectRef) -> CString;
    #[name("node.StructuralHash")]
    fn structural_hash(object: ObjectRef, map_free_vars: bool) -> i64;
    #[name("node.StructuralEqual")]
    fn structural_equal(lhs: ObjectRef, rhs: ObjectRef, assert_mode: bool, map_free_vars: bool) -> bool;
}
