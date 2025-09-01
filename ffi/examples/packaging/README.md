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

# TVM FFI Packaging Example

This is an example project that packages a tvm-ffi based library
into a Python ABI-agnostic wheel.

This example can also serve as a guideline for general
packaging as well.

- Source-level build for cross-compilation support in CMake
- Registration via global function table

## Install the wheel

```bash
pip install .
```

### Note on build and auditwheel

Note: When running the auditwheel process, make sure to skip
`libtvm_ffi.so` as they are shipped via the tvm_ffi package.

## Run the example

After installing the `my_ffi_extension` example package, you can run the following example
that invokes the `add_one` function exposed.

```bash
python run_example.py add_one
```

You can also run the following command to see how error is raised and propagated
across the language boundaries.

```python
python run_example.py raise_error
```

When possible, tvm_ffi will try to preserve traceback across language boundary. You will see traceback like
```
File "src/extension.cc", line 45, in void my_ffi_extension::RaiseError(tvm::ffi::String)
```
If you are in an IDE like VSCode, you can click and jump to the C++ lines of error when
the debug symbols are preserved.
