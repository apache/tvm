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
use std::iter::{FromIterator, IntoIterator, Iterator};
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
    #[name("runtime.ArrayGetItem")]
    fn array_get_item(array: ObjectRef, index: isize) -> ObjectRef;
    #[name("runtime.ArraySize")]
    fn array_size(array: ObjectRef) -> i64;
}

impl<T: IsObjectRef + 'static> IsObjectRef for Array<T> {
    type Object = Object;
    fn as_ptr(&self) -> Option<&ObjectPtr<Self::Object>> {
        self.object.as_ptr()
    }

    fn into_ptr(self) -> Option<ObjectPtr<Self::Object>> {
        self.object.into_ptr()
    }

    fn from_ptr(object_ptr: Option<ObjectPtr<Self::Object>>) -> Self {
        let object_ref = match object_ptr {
            Some(o) => o.into(),
            _ => panic!(),
        };

        Array {
            object: object_ref,
            _data: PhantomData,
        }
    }
}

impl<T: IsObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        let iter = data.iter().map(T::into_arg_value).collect();

        let func = Function::get("runtime.Array").expect(
            "runtime.Array function is not registered, this is most likely a build or linking error",
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

impl<T: IsObjectRef> std::fmt::Debug for Array<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let as_vec: Vec<T> = self.clone().into_iter().collect();
        write!(formatter, "{:?}", as_vec)
    }
}

pub struct IntoIter<T: IsObjectRef> {
    array: Array<T>,
    pos: isize,
    size: isize,
}

impl<T: IsObjectRef> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.size {
            let item =
                self.array.get(self.pos)
                    .expect("Can not index as in-bounds position after bounds checking.\nNote: this error can only be do to an uncaught issue with API bindings.");
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<T: IsObjectRef> IntoIterator for Array<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let size = self.len() as isize;
        IntoIter {
            array: self,
            pos: 0,
            size: size,
        }
    }
}

impl<T: IsObjectRef> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Array::from_vec(iter.into_iter().collect()).unwrap()
    }
}

impl<'a, T: IsObjectRef> From<&'a Array<T>> for ArgValue<'a> {
    fn from(array: &'a Array<T>) -> ArgValue<'a> {
        (&array.object).into()
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

#[cfg(test)]
mod tests {
    use super::Array;
    use crate::function::Result;
    use crate::object::{IsObjectRef, ObjectRef};
    use crate::string::String;

    #[test]
    fn create_array_and_get() -> Result<()> {
        let vec: Vec<String> = vec!["foo".into(), "bar".into(), "baz".into()];
        let array = Array::from_vec(vec)?;
        assert_eq!(array.get(0)?.to_string(), "foo");
        assert_eq!(array.get(1)?.to_string(), "bar");
        assert_eq!(array.get(2)?.to_string(), "baz");
        Ok(())
    }

    #[test]
    fn downcast() -> Result<()> {
        let vec: Vec<String> = vec!["foo".into(), "bar".into(), "baz".into()];
        let array: ObjectRef = ObjectRef::from_ptr(Array::from_vec(vec)?.into_ptr());
        let array: Array<ObjectRef> = array.downcast::<Array<ObjectRef>>().unwrap();
        assert_eq!(array.get(1)?.downcast::<String>().unwrap(), "bar");
        Ok(())
    }
}
