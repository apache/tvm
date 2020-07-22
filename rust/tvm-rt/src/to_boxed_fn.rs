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

//! This module provides a method for converting type erased TVM functions
//! into a boxed Rust closure.
//!
//! To call a registered function check the [`ToBoxedFn::to_boxed_fn`] method.
//!
//! See the tests and examples repository for more examples.

pub use tvm_sys::{ffi, ArgValue, RetValue};

use crate::errors;

use super::function::{Function, Result};

pub trait ToBoxedFn {
    fn to_boxed_fn(func: Function) -> Box<Self>;
}

use std::convert::{TryFrom, TryInto};

impl<E, O> ToBoxedFn for dyn Fn() -> Result<O>
where
    errors::Error: From<E>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: Function) -> Box<Self> {
        Box::new(move || {
            let res = func.invoke(vec![])?;
            let res = res.try_into()?;
            Ok(res)
        })
    }
}

impl<E, A, O> ToBoxedFn for dyn Fn(A) -> Result<O>
where
    errors::Error: From<E>,
    A: Into<ArgValue<'static>>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: Function) -> Box<Self> {
        Box::new(move |a: A| {
            let args = vec![a.into()];
            let res = func.invoke(args)?;
            let res = res.try_into()?;
            Ok(res)
        })
    }
}

impl<E, A, B, O> ToBoxedFn for dyn Fn(A, B) -> Result<O>
where
    errors::Error: From<E>,
    A: Into<ArgValue<'static>>,
    B: Into<ArgValue<'static>>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: Function) -> Box<Self> {
        Box::new(move |a: A, b: B| {
            let args = vec![a.into(), b.into()];
            let res = func.invoke(args)?;
            let res = res.try_into()?;
            Ok(res)
        })
    }
}

impl<E, A, B, C, O> ToBoxedFn for dyn Fn(A, B, C) -> Result<O>
where
    errors::Error: From<E>,
    A: Into<ArgValue<'static>>,
    B: Into<ArgValue<'static>>,
    C: Into<ArgValue<'static>>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: Function) -> Box<Self> {
        Box::new(move |a: A, b: B, c: C| {
            let args = vec![a.into(), b.into(), c.into()];
            let res = func.invoke(args)?;
            let res = res.try_into()?;
            Ok(res)
        })
    }
}

impl<E, A, B, C, D, O> ToBoxedFn for dyn Fn(A, B, C, D) -> Result<O>
where
    errors::Error: From<E>,
    A: Into<ArgValue<'static>>,
    B: Into<ArgValue<'static>>,
    C: Into<ArgValue<'static>>,
    D: Into<ArgValue<'static>>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: Function) -> Box<Self> {
        Box::new(move |a: A, b: B, c: C, d: D| {
            let args = vec![a.into(), b.into(), c.into(), d.into()];
            let res = func.invoke(args)?;
            let res = res.try_into()?;
            Ok(res)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::function::{self, Function, Result};

    #[test]
    fn to_boxed_fn0() {
        fn boxed0() -> i64 {
            return 10;
        }

        function::register_override(boxed0, "boxed0".to_owned(), true).unwrap();
        let func = Function::get("boxed0").unwrap();
        let typed_func: Box<dyn Fn() -> Result<i64>> = func.to_boxed_fn();
        assert_eq!(typed_func().unwrap(), 10);
    }
}
