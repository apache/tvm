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

//! This module implements [`TVMArgValue`] and [`TVMRetValue`] types
//! and their conversions needed for the types used in frontend crate.
//! `TVMRetValue` is the owned version of `TVMPODValue`.

use std::convert::TryFrom;

use tvm_common::{
    errors::ValueDowncastError,
    ffi::{TVMArrayHandle, TVMFunctionHandle, TVMModuleHandle},
    try_downcast,
};

use crate::{Function, Module, NDArray, TVMArgValue, TVMRetValue};

macro_rules! impl_handle_val {
    ($type:ty, $variant:ident, $inner_type:ty, $ctor:path) => {
        impl<'a> From<&'a $type> for TVMArgValue<'a> {
            fn from(arg: &'a $type) -> Self {
                TVMArgValue::$variant(arg.handle() as $inner_type)
            }
        }

        impl<'a> From<&'a mut $type> for TVMArgValue<'a> {
            fn from(arg: &'a mut $type) -> Self {
                TVMArgValue::$variant(arg.handle() as $inner_type)
            }
        }

        impl<'a> TryFrom<TVMArgValue<'a>> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: TVMArgValue<'a>) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |TVMArgValue::$variant(val)| { $ctor(val) })
            }
        }

        impl<'a, 'v> TryFrom<&'a TVMArgValue<'v>> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: &'a TVMArgValue<'v>) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |TVMArgValue::$variant(val)| { $ctor(*val) })
            }
        }

        impl From<$type> for TVMRetValue {
            fn from(val: $type) -> TVMRetValue {
                TVMRetValue::$variant(val.handle() as $inner_type)
            }
        }

        impl TryFrom<TVMRetValue> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: TVMRetValue) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |TVMRetValue::$variant(val)| { $ctor(val) })
            }
        }
    };
}

impl_handle_val!(Function, FuncHandle, TVMFunctionHandle, Function::new);
impl_handle_val!(Module, ModuleHandle, TVMModuleHandle, Module::new);
impl_handle_val!(NDArray, ArrayHandle, TVMArrayHandle, NDArray::new);

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, str::FromStr};

    use tvm_common::{TVMByteArray, TVMContext, TVMType};

    use super::*;

    #[test]
    fn bytearray() {
        let w = vec![1u8, 2, 3, 4, 5];
        let v = TVMByteArray::from(w.as_slice());
        let tvm: TVMByteArray = TVMRetValue::from(v).try_into().unwrap();
        assert_eq!(
            tvm.data(),
            w.iter().map(|e| *e).collect::<Vec<u8>>().as_slice()
        );
    }

    #[test]
    fn ty() {
        let t = TVMType::from_str("int32").unwrap();
        let tvm: TVMType = TVMRetValue::from(t).try_into().unwrap();
        assert_eq!(tvm, t);
    }

    #[test]
    fn ctx() {
        let c = TVMContext::from_str("gpu").unwrap();
        let tvm: TVMContext = TVMRetValue::from(c).try_into().unwrap();
        assert_eq!(tvm, c);
    }
}
