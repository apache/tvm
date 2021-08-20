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

use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::iter::FromIterator;
use std::marker::PhantomData;

use crate::object::debug_print;

use crate::array::Array;
use crate::errors::Error;
use crate::object::{IsObjectRef, Object, ObjectPtr, ObjectRef};
use crate::ArgValue;
use crate::{
    external,
    function::{Function, Result},
    RetValue,
};

#[repr(C)]
#[derive(Clone)]
pub struct Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    object: ObjectRef,
    _data: PhantomData<(K, V)>,
}

// TODO(@jroesch): convert to use generics instead of casting inside
// the implementation.
external! {
   #[name("runtime.MapSize")]
   fn map_size(map: ObjectRef) -> i64;
   #[name("runtime.MapGetItem")]
   fn map_get_item(map_object: ObjectRef, key: ObjectRef) -> ObjectRef;
   #[name("runtime.MapCount")]
   fn map_count(map: ObjectRef, key: ObjectRef) -> ObjectRef;
   #[name("runtime.MapItems")]
   fn map_items(map: ObjectRef) -> Array<ObjectRef>;
}

impl<'a, K: 'a, V: 'a> FromIterator<(&'a K, &'a V)> for Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    fn from_iter<T: IntoIterator<Item = (&'a K, &'a V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, upper_bound) = iter.size_hint();
        let mut buffer: Vec<ArgValue> = Vec::with_capacity(upper_bound.unwrap_or(lower_bound) * 2);
        for (k, v) in iter {
            buffer.push(k.into_arg_value());
            buffer.push(v.into_arg_value());
        }
        Self::from_data(buffer).expect("failed to convert from data")
    }
}

impl<K, V> Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    pub fn from_data(data: Vec<ArgValue>) -> Result<Map<K, V>> {
        let func = Function::get("runtime.Map").expect(
            "runtime.Map function is not registered, this is most likely a build or linking error",
        );

        let map_data: ObjectPtr<Object> = func.invoke(data)?.try_into()?;

        debug_assert!(
            map_data.count() >= 1,
            "map_data count is {}",
            map_data.count()
        );

        Ok(Map {
            object: map_data.into(),
            _data: PhantomData,
        })
    }

    pub fn get(&self, key: &K) -> Result<V>
    where
        V: TryFrom<RetValue, Error = Error>,
    {
        let key = key.clone();
        let oref: ObjectRef = map_get_item(self.object.clone(), key.upcast())?;
        oref.downcast()
    }

    pub fn empty() -> Self {
        Self::from_iter(vec![].into_iter())
    }

    //(@jroesch): I don't think this is a correct implementation.
    pub fn null() -> Self {
        Map {
            object: ObjectRef::null(),
            _data: PhantomData,
        }
    }
}

pub struct IntoIter<K, V> {
    // NB: due to FFI this isn't as lazy as one might like
    key_and_values: Array<ObjectRef>,
    next_key: i64,
    _data: PhantomData<(K, V)>,
}

impl<K, V> Iterator for IntoIter<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        if self.next_key < self.key_and_values.len() {
            let key = self
                .key_and_values
                .get(self.next_key as isize)
                .expect("this should always succeed");
            let value = self
                .key_and_values
                .get((self.next_key as isize) + 1)
                .expect("this should always succeed");
            self.next_key += 2;
            Some((key.downcast().unwrap(), value.downcast().unwrap()))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        ((self.key_and_values.len() / 2) as usize, None)
    }
}

impl<K, V> IntoIterator for Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        let items = map_items(self.object).expect("unable to get map items");
        IntoIter {
            key_and_values: items,
            next_key: 0,
            _data: PhantomData,
        }
    }
}

use std::fmt;

impl<K, V> fmt::Debug for Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctr = debug_print(self.object.clone()).unwrap();
        fmt.write_fmt(format_args!("{:?}", ctr))
    }
}

impl<K, V, S> From<Map<K, V>> for HashMap<K, V, S>
where
    K: Eq + std::hash::Hash,
    K: IsObjectRef,
    V: IsObjectRef,
    S: std::hash::BuildHasher + std::default::Default,
{
    fn from(map: Map<K, V>) -> HashMap<K, V, S> {
        HashMap::from_iter(map.into_iter())
    }
}

impl<'a, K, V> From<&'a Map<K, V>> for ArgValue<'a>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    fn from(map: &'a Map<K, V>) -> ArgValue<'a> {
        (&map.object).into()
    }
}

impl<K, V> From<Map<K, V>> for RetValue
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    fn from(map: Map<K, V>) -> RetValue {
        map.object.into()
    }
}

impl<'a, K, V> TryFrom<ArgValue<'a>> for Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    type Error = Error;

    fn try_from(array: ArgValue<'a>) -> Result<Map<K, V>> {
        let object_ref: ObjectRef = array.try_into()?;
        // TODO: type check
        Ok(Map {
            object: object_ref,
            _data: PhantomData,
        })
    }
}

impl<K, V> TryFrom<RetValue> for Map<K, V>
where
    K: IsObjectRef,
    V: IsObjectRef,
{
    type Error = Error;

    fn try_from(array: RetValue) -> Result<Map<K, V>> {
        let object_ref = array.try_into()?;
        // TODO: type check
        Ok(Map {
            object: object_ref,
            _data: PhantomData,
        })
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::*;
    use crate::string::String as TString;

    #[test]
    fn test_from_into_hash_map() {
        let mut std_map: HashMap<TString, TString> = HashMap::new();
        std_map.insert("key1".into(), "value1".into());
        std_map.insert("key2".into(), "value2".into());
        let tvm_map = Map::from_iter(std_map.iter());
        let back_map = tvm_map.into();
        assert_eq!(std_map, back_map);
    }
}
