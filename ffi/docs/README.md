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
# TVM FFI Documentation

To build locally

First install the tvm-ffi package
```bash
pip install ..
```

Install all the requirements to build docs

```bash
pip install -r requirements.txt
```

Then build the doc
```bash
make livehtml
```

## Build with C++ Docs

To build with C++ docs, we need to first install Doxygen. Then
set the environment variable `BUILD_CPP_DOCS=1`, to turn on c++ docs.

```bash
BUILD_CPP_DOCS=1 make livehtml
```

Building c++ docs can take longer, so it is not on by default.
