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

## Resnet example

This end-to-end example shows how to:
* build `Resnet 18` with `tvm` from Python
* use the provided Rust frontend API to test for an input image

To run the example with pretrained resnet weights, first `tvm`  and `mxnet` must be installed for the python build. To install mxnet for cpu, run `pip install mxnet`
and to install `tvm` with `llvm` follow the [TVM installation guide](https://tvm.apache.org/docs/install/index.html).

* **Build the example**: `cargo build

To have a successful build, note that it is required to instruct Rust compiler to link to the compiled shared library, for example with
`println!("cargo:rustc-link-search=native={}", build_path)`. See the `build.rs` for more details.

* **Run the example**: `cargo run`

Note: To use pretrained weights, one can enable `--pretrained` in `build.rs` with

```
let output = Command::new("python")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/build_resnet.py"))
        .arg(&format!("--build-dir={}", env!("CARGO_MANIFEST_DIR")))
        .arg(&format!("--pretrained"))
        .output()
        .expect("Failed to execute command");
```

Otherwise, *random weights* are used, therefore, the prediction will be `limpkin, Aramus pictus`!
