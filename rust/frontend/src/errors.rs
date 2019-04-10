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

pub use failure::Error;

#[derive(Debug, Fail)]
#[fail(display = "Cannot convert from an empty array.")]
pub struct EmptyArrayError;

#[derive(Debug, Fail)]
#[fail(display = "Handle `{}` is null.", name)]
pub struct NullHandleError {
    pub name: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Function was not set in `function::Builder`")]
pub struct FunctionNotFoundError;

#[derive(Debug, Fail)]
#[fail(display = "Expected type `{}` but found `{}`", expected, actual)]
pub struct TypeMismatchError {
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Fail)]
#[fail(display = "Missing NDArray shape.")]
pub struct MissingShapeError;
