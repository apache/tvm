/*!
 *  Copyright (c) 2018 by Contributors
 * \brief This is an all in one TVM runtime file.
 * \file tvm.cc
 */
#include "src/runtime/c_runtime_api.cc"
#include "src/runtime/cpu_device_api.cc"
#include "src/runtime/workspace_pool.cc"
#include "src/runtime/module_util.cc"
#include "src/runtime/module.cc"
#include "src/runtime/registry.cc"
#include "src/runtime/file_util.cc"
#include "src/runtime/threading_backend.cc"
#include "src/runtime/thread_pool.cc"
#include "src/runtime/ndarray.cc"

// NOTE: all the files after this are optional modules
// that you can include remove, depending on how much feature you use.

// Likely we only need to enable one of the following
// If you use Module::Load, use dso_module
// For system packed library, use system_lib_module
#include "src/runtime/dso_module.cc"
#include "src/runtime/system_lib_module.cc"

// Graph runtime
#include "src/runtime/graph/graph_runtime.cc"

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
