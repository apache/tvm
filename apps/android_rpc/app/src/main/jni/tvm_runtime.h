/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm_runtime.h
 * \brief Pack all tvm runtime source files
 */
#include <sys/stat.h>
#include <fstream>

#define DMLC_LOG_CUSTOMIZE 1
#define DMLC_LOG_BEFORE_THROW 1

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/workspace_pool.cc"
#include "../src/runtime/module_util.cc"
#include "../src/runtime/system_lib_module.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/file_util.cc"
#include "../src/runtime/dso_module.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/rpc/rpc_module.cc"
#include "../src/runtime/rpc/rpc_socket_impl.cc"
#include "../src/runtime/thread_pool.cc"
#include "../src/runtime/threading_backend.cc"
#include "../src/runtime/graph/graph_runtime.cc"
#include "../src/runtime/ndarray.cc"

#ifdef TVM_OPENCL_RUNTIME
#include "../src/runtime/opencl/opencl_device_api.cc"
#include "../src/runtime/opencl/opencl_module.cc"
#endif

#ifdef TVM_VULKAN_RUNTIME
#include "../src/runtime/vulkan/vulkan_device_api.cc"
#include "../src/runtime/vulkan/vulkan_module.cc"
#endif


#include <android/log.h>

void dmlc::CustomLogMessage::Log(const std::string& msg) {
  __android_log_print(ANDROID_LOG_DEBUG, "TVM_RUNTIME", "%s", msg.c_str());
}
