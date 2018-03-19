/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx_runtime.cc
 */
#include "../../src/runtime/c_runtime_api.cc"
#include "../../src/runtime/cpu_device_api.cc"
#include "../../src/runtime/workspace_pool.cc"
#include "../../src/runtime/module_util.cc"
#include "../../src/runtime/module.cc"
#include "../../src/runtime/registry.cc"
#include "../../src/runtime/system_lib_module.cc"
#ifndef _LIBCPP_SGX_CONFIG
#include "../../src/runtime/threading_backend.cc"
#else
#include "threading_backend.cc"
#endif
#include "../../src/runtime/thread_pool.cc"
