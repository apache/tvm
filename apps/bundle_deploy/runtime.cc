#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include "../../src/runtime/c_runtime_api.cc"
#include "../../src/runtime/cpu_device_api.cc"
#include "../../src/runtime/workspace_pool.cc"
#include "../../src/runtime/module_util.cc"
#include "../../src/runtime/module.cc"
#include "../../src/runtime/registry.cc"
#include "../../src/runtime/file_util.cc"
#include "../../src/runtime/threading_backend.cc"
#include "../../src/runtime/thread_pool.cc"
#include "../../src/runtime/ndarray.cc"
#include "../../src/runtime/system_lib_module.cc"
#include "../../src/runtime/graph/graph_runtime.cc"
