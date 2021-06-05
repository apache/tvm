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

/*!
 * \brief This is an all in one TVM runtime file.
 * \file tvm_runtime_pack.cc
 */
#include "src/runtime/c_runtime_api.cc"
#include "src/runtime/container.cc"
#include "src/runtime/cpu_device_api.cc"
#include "src/runtime/file_utils.cc"
#include "src/runtime/library_module.cc"
#include "src/runtime/logging.cc"
#include "src/runtime/module.cc"
#include "src/runtime/ndarray.cc"
#include "src/runtime/object.cc"
#include "src/runtime/registry.cc"
#include "src/runtime/thread_pool.cc"
#include "src/runtime/threading_backend.cc"
#include "src/runtime/workspace_pool.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "src/runtime/dso_library.cc"
#include "src/runtime/system_library.cc"

// Graph executor
#include "src/runtime/graph_executor/graph_executor.cc"

// Uncomment the following lines to enable RPC
// #include "../../src/runtime/rpc/rpc_session.cc"
// #include "../../src/runtime/rpc/rpc_event_impl.cc"
// #include "../../src/runtime/rpc/rpc_server_env.cc"

// These macros enables the device API when uncommented.
#define TVM_CUDA_RUNTIME 1
#define TVM_METAL_RUNTIME 1
#define TVM_OPENCL_RUNTIME 1

// Uncomment the following lines to enable Metal
// #include "../../src/runtime/metal/metal_device_api.mm"
// #include "../../src/runtime/metal/metal_module.mm"

// Uncomment the following lines to enable CUDA
// #include "../../src/runtime/cuda/cuda_device_api.cc"
// #include "../../src/runtime/cuda/cuda_module.cc"

// Uncomment the following lines to enable OpenCL
// #include "../../src/runtime/opencl/opencl_device_api.cc"
// #include "../../src/runtime/opencl/opencl_module.cc"
// #include "../src/runtime/source_utils.cc"
