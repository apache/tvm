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

//! This module implements [`ArgValue`] and [`RetValue`] types
//! and their conversions needed for the types used in frontend crate.
//! `RetValue` is the owned version of `TVMPODValue`.

use std::convert::TryFrom;

use crate::{ArgValue, Module, RetValue};
use tvm_sys::{errors::ValueDowncastError, ffi::TVMModuleHandle, try_downcast};

macro_rules! impl_handle_val {
    ($type:ty, $variant:ident, $inner_type:ty, $ctor:path) => {
        impl<'a> From<&'a $type> for ArgValue<'a> {
            fn from(arg: &'a $type) -> Self {
                ArgValue::$variant(arg.handle() as $inner_type)
            }
        }

        impl<'a> From<&'a mut $type> for ArgValue<'a> {
            fn from(arg: &'a mut $type) -> Self {
                ArgValue::$variant(arg.handle() as $inner_type)
            }
        }

        impl<'a> TryFrom<ArgValue<'a>> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: ArgValue<'a>) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |ArgValue::$variant(val)| { $ctor(val) })
            }
        }

        impl<'a, 'v> TryFrom<&'a ArgValue<'v>> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: &'a ArgValue<'v>) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |ArgValue::$variant(val)| { $ctor(*val) })
            }
        }

        impl From<$type> for RetValue {
            fn from(val: $type) -> RetValue {
                RetValue::$variant(val.handle() as $inner_type)
            }
        }

        impl TryFrom<RetValue> for $type {
            type Error = ValueDowncastError;
            fn try_from(val: RetValue) -> Result<$type, Self::Error> {
                try_downcast!(val -> $type, |RetValue::$variant(val)| { $ctor(val) })
            }
        }
    };
}

impl_handle_val!(Module, ModuleHandle, TVMModuleHandle, Module::new);

#[cfg(test)]
mod tests {
    use std::{convert::TryInto, str::FromStr};

    use crate::{ByteArray, Context, DataType};

    use super::*;

    #[test]
    fn bytearray() {
        let w = vec![1u8, 2, 3, 4, 5];
        let v = ByteArray::from(w.as_slice());
        let tvm: ByteArray = RetValue::from(v).try_into().unwrap();
        assert_eq!(
            tvm.data(),
            w.iter().copied().collect::<Vec<u8>>().as_slice()
        );
    }

    #[test]
    fn ty() {
        let t = DataType::from_str("int32").unwrap();
        let tvm: DataType = RetValue::from(t).try_into().unwrap();
        assert_eq!(tvm, t);
    }

    #[test]
    fn ctx() {
        let c = Context::from_str("gpu").unwrap();
        let tvm: Context = RetValue::from(c).try_into().unwrap();
        assert_eq!(tvm, c);
    }
}
