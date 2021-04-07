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

# gotvm - Golang Frontend for TVM Runtime

This folder contain golang interface for TVM runtime. It brings TVM runtime to Golang.

- It enable c runtime api of tvm exposed to golang.
- It enables module loading (lib, graph and params) and inference operations.

## Installation

### Requirements

- go compiler (https://golang.org/) version 0.10 or above.

### Modules

- src
  Module that generates golang package corresponding to the c runtime api exposed from tvm source tree.
  This process build golang package _gotvm.a_

- samples
  Sample golang reference application to inference through gotvm package.

### Build

Once the Requirements are installed

To build _gotvm_ package

```bash
make
```

To build and run internal tests

```bash
make tests
```

To build sample apps.

```bash
make samples
```

## Run

To Demonstrates sample TVM module compilation using python and deploy via golang.
```bash
./simple
```

To deploy a realtime module with lib, graph and param.
```bash
python3 gen_mobilenet_lib.py

./complex
```

To demonstrate go function closure conversion to packed function handle.

```bash
./pack_func_convert
```

To demonstrate a packed function handle given as an argument.

```bash
./pack_func_handle_arg
```

To register go function with runtime as a global function.

```bash
./pack_func_register
```

To demonstrate function closure passed as argument to a function call.

```bash
./pack_func_closure_arg
```

To demonstrate function closure returned from a packed function.

```bash
./pack_func_closure_return
```

## Documentation
gotvm.go is documented with sufficient information about gotvm package.
A html version documentation can be accessed by running below command after building runtime.

```bash
godoc -http=:6060  -goroot=./gopath
```
After above command try http://127.0.0.1:6060 from any browser.

Also please refer to the sample applications under sample folder.

## Docker
Docker setup may need below additions for dependencies and environment preparation.

Please refer ```docker/install/ubuntu_install_golang.sh``` for the packages dependencies.

go compiler 1.10 on ubuntu doesn't install on standard path, hence an explicit export may be needed as shown below.

```bash
export PATH="/usr/lib/go-1.10/bin:$PATH"
```
