<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM Runtime Support

This crate provides an idiomatic Rust API for [TVM](https://github.com/apache/tvm) runtime,
see [here](https://github.com/apache/tvm/blob/main/rust/tvm/README.md) for more details.

## What Does This Crate Offer?

TVM is an end-to-end deep learning compiler which takes high level machine learning
models or tensor computations and lowers them into executable code for a variety
of heterogenous devices (e.g., CPU, GPU).

This crate provides access to the APIs for manipulating runtime data structures,
as well as TVM's cross-language Object system which functions similarly to systems
such as COM, enabling cross-language interoperability.

## Installations

Please follow TVM [installation](https://tvm.apache.org/docs/install/index.html) instructions,
`export TVM_HOME=/path/to/tvm` and add `libtvm_runtime` to your `LD_LIBRARY_PATH`.

### Example of registering a cross-language closure.

One can use `register!` macro to expose a Rust closure with arguments which implement `TryFrom<ArgValue>`
and return types which implement `Into<RetValue>`. Once registered with TVM these functions can be
accessed via Python or C++, or any other language which implements the TVM packed function convention
see the offcial documentation for more information.

```rust
use tvm_rt::{ArgValue, RetValue};
use tvm_rt::function::{Function, Result, register};

fn sum(x: i64, y: i64, z: i64) -> i64 {
    x + y + z
}

fn main() {
    register(sum, "mysum".to_owned()).unwrap();
    let func = Function::get("mysum").unwrap();
    let boxed_fn: Box<dyn Fn(i64, i64, i64) -> Result<i64>> = func.into();
    let ret = boxed_fn(10, 20, 30).unwrap();
    assert_eq!(ret, 60);
}
```
