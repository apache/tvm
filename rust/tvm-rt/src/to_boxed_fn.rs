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

//! This module provides an idiomatic Rust API for creating and working with TVM functions.
//!
//! For calling an already registered TVM function use [`function::Builder`]
//! To register a TVM packed function from Rust side either
//! use [`function::register`] or the macro [`register_global_func`].
//!
//! See the tests and examples repository for more examples.

pub use tvm_sys::{ffi, ArgValue, RetValue};

use crate::{Module, errors};

use super::function::{Function, Result};

pub trait ToBoxedFn {
    fn to_boxed_fn(func: &'static Function) -> Box<Self>;
}

use std::convert::{TryFrom, TryInto};

impl<E, O> ToBoxedFn for dyn Fn() -> Result<O>
where
    errors::Error: From<E>,
    O: TryFrom<RetValue, Error = E>,
{
    fn to_boxed_fn(func: &'static Function) -> Box<Self> {
        Box::new(move || {
            let mut builder = Builder::default();
            builder.func = Some(func);
            let res = builder.invoke()?.try_into()?;
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
    fn to_boxed_fn(func: &'static Function) -> Box<Self> {
        Box::new(move |a: A| {
            let mut builder = Builder::default();
            builder.func = Some(func);
            builder.arg(a.into());
            let res = builder.invoke()?.try_into()?;
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
    fn to_boxed_fn(func: &'static Function) -> Box<Self> {
        Box::new(move |a: A, b: B| {
            let mut builder = Builder::default();
            builder.func = Some(func);
            builder.arg(a.into());
            builder.arg(b.into());
            let res = builder.invoke()?.try_into()?;
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
    fn to_boxed_fn(func: &'static Function) -> Box<Self> {
        Box::new(move |a: A, b: B, c: C| {
            let mut builder = Builder::default();
            builder.func = Some(func);
            builder.arg(a.into());
            builder.arg(b.into());
            builder.arg(c.into());
            let res = builder.invoke()?.try_into()?;
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
    fn to_boxed_fn(func: &'static Function) -> Box<Self> {
        Box::new(move |a: A, b: B, c: C, d: D| {
            let mut builder = Builder::default();
            builder.func = Some(func);
            builder.arg(a.into());
            builder.arg(b.into());
            builder.arg(c.into());
            builder.arg(d.into());
            let res = builder.invoke()?.try_into()?;
            Ok(res)
        })
    }
}

/// Function builder in order to create and call functions.
///
/// *Note:* Currently TVM functions accept *at most* one return value.
#[derive(Default)]
pub struct Builder<'a, 'm> {
    pub func: Option<&'m Function>,
    pub arg_buf: Vec<ArgValue<'a>>,
    pub ret_buf: Option<RetValue>,
}

impl<'a, 'm> Builder<'a, 'm> {
    pub fn new(
        func: Option<&'m Function>,
        arg_buf: Vec<ArgValue<'a>>,
        ret_buf: Option<RetValue>,
    ) -> Self {
        Self {
            func,
            arg_buf,
            ret_buf,
        }
    }

    pub fn get_function(&mut self, name: &'m str) -> &mut Self {
        self.func = Function::get(name);
        self
    }

    /// Pushes a [`ArgValue`] into the function argument buffer.
    pub fn arg<T: 'a>(&mut self, arg: T) -> &mut Self
    where
        ArgValue<'a>: From<T>,
    {
        self.arg_buf.push(arg.into());
        self
    }

    /// Pushes multiple [`ArgValue`]s into the function argument buffer.
    pub fn args<T: 'a, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = T>,
        ArgValue<'a>: From<T>,
    {
        args.into_iter().for_each(|arg| {
            self.arg(arg);
        });
        self
    }

    /// Sets an output for a function that requires a mutable output to be provided.
    /// See the `basics` in tests for an example.
    pub fn set_output<T>(&mut self, ret: T) -> &mut Self
    where
        RetValue: From<T>,
    {
        self.ret_buf = Some(ret.into());
        self
    }

    pub fn invoke(self) -> Result<RetValue> {
        self.func.unwrap().invoke(self.arg_buf)
    }
}

/// Converts a [`Function`] to builder. Currently, this is the best way to work with
/// TVM functions.
impl<'a, 'm> From<&'m Function> for Builder<'a, 'm> {
    fn from(func: &'m Function) -> Self {
        Builder::new(Some(func), Vec::new(), None)
    }
}

/// Converts a mutable reference of a [`Module`] to [`Builder`].
impl<'a, 'm> From<&'m mut Module> for Builder<'a, 'm> {
    fn from(module: &'m mut Module) -> Self {
        Builder::new(module.entry(), Vec::new(), None)
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
