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
    RetValue,
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
}

impl<T: IsObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        let iter = data
            .iter()
            .map(|element| element.to_object_ref().into())
            .collect();

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
            object: ObjectRef(Some(array_data)),
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
}
