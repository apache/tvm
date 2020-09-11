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

use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use crate::errors::Error;
use crate::object::{IsObjectRef, Object, ObjectPtr, ObjectRef};
use crate::{
    external,
    function::{Function, Result},
    ArgValue, RetValue,
};

#[repr(C)]
#[derive(Clone)]
pub struct Array<T: IsObjectRef> {
    object: ObjectRef,
    _data: PhantomData<T>,
}

// TODO(@jroesch): convert to use generics instead of casting inside
// the implementation.
external! {
    #[name("node.ArrayGetItem")]
    fn array_get_item(array: ObjectRef, index: isize) -> ObjectRef;
    #[name("node.ArraySize")]
    fn array_size(array: ObjectRef) -> i64;
}

impl<T: IsObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        let iter = data.into_iter().map(T::into_arg_value).collect();

        let func = Function::get("node.Array").expect(
            "node.Array function is not registered, this is most likely a build or linking error",
        );

        // let array_data = func.invoke(iter)?;
        // let array_data: ObjectRef = func.invoke(iter)?.try_into()?;
        let array_data: ObjectPtr<Object> = func.invoke(iter)?.try_into()?;

        debug_assert!(
            array_data.count() >= 1,
            "array reference count is {}",
            array_data.count()
        );

        Ok(Array {
            object: array_data.into(),
            _data: PhantomData,
        })
    }

    pub fn get(&self, index: isize) -> Result<T>
    where
        T: TryFrom<RetValue, Error = Error>,
    {
        let oref: ObjectRef = array_get_item(self.object.clone(), index)?;
        oref.downcast()
    }

    pub fn len(&self) -> i64 {
        array_size(self.object.clone()).expect("size should never fail")
    }
}

impl<T: IsObjectRef> From<Array<T>> for ArgValue<'static> {
    fn from(array: Array<T>) -> ArgValue<'static> {
        array.object.into()
    }
}

impl<T: IsObjectRef> From<Array<T>> for RetValue {
    fn from(array: Array<T>) -> RetValue {
        array.object.into()
    }
}

impl<'a, T: IsObjectRef> TryFrom<ArgValue<'a>> for Array<T> {
    type Error = Error;

    fn try_from(array: ArgValue<'a>) -> Result<Array<T>> {
        let object_ref: ObjectRef = array.try_into()?;
        // TODO: type check
        Ok(Array {
            object: object_ref,
            _data: PhantomData,
        })
    }
}

impl<'a, T: IsObjectRef> TryFrom<RetValue> for Array<T> {
    type Error = Error;

    fn try_from(array: RetValue) -> Result<Array<T>> {
        let object_ref = array.try_into()?;
        Ok(Array {
            object: object_ref,
            _data: PhantomData,
        })
    }
}
