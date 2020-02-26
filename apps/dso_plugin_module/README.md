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


Example Plugin Module
=====================
This folder contains an example that implements a C++ module
that can be directly loaded as TVM's DSOModule (via tvm.runtime.load_module)

## Guideline

When possible, we always recommend exposing
functions that modifies memory passed by the caller,
and calls into the runtime API for memory allocations.

## Advanced Usecases

In advanced usecases, we do allow the plugin module to
create and return managed objects.
However, there are several restrictions to keep in mind:

- If the module returns an object, we need to make sure
  that the object get destructed before the module get unloaded.
  Otherwise segfault can happen because of calling into an unloaded destructor.
- If the module returns a PackedFunc, then
  we need to ensure that the libc of the DLL and tvm runtime matches.
  Otherwise segfault can happen due to incompatibility of std::function.
