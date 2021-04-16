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

# tvm-sys

The low level bindings to TVM's C APIs for interacting with the runtime,
the cross-language object system, and packed function API.

These will generate bindings to TVM, if you set `TVM_HOME` variable before
building it will instruct the bindings to use your source tree, if not the
crate will use `tvm-build` in order to build a sandboxed version of the library.

This feature is intended to simplify the installation for brand new TVM users
by trying to automate the build process as much as possible.
