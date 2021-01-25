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
 * \file tvm_runtime.h
 * \brief Pack all tvm runtime source files
 */
#include <sys/stat.h>

#include <fstream>

/* Enable custom logging - this will cause TVM to use a custom implementation
 * of tvm::runtime::detail::LogMessage. We use this to pass TVM log messages to
 * Android logcat.
 */
#define TVM_LOG_CUSTOMIZE 1

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/dso_library.cc"
#include "../src/runtime/file_utils.cc"
#include "../src/runtime/graph/graph_runtime.cc"
#include "../src/runtime/graph/graph_runtime_factory.cc"
#include "../src/runtime/library_module.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/ndarray.cc"
#include "../src/runtime/object.cc"
#include "../src/runtime/profiling.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/rpc/rpc_channel.cc"
#include "../src/runtime/rpc/rpc_endpoint.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_local_session.cc"
#include "../src/runtime/rpc/rpc_module.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_socket_impl.cc"
#include "../src/runtime/system_library.cc"
#include "../src/runtime/thread_pool.cc"
#include "../src/runtime/threading_backend.cc"
#include "../src/runtime/workspace_pool.cc"

#ifdef TVM_OPENCL_RUNTIME
#include "../src/runtime/opencl/opencl_device_api.cc"
#include "../src/runtime/opencl/opencl_module.cc"
#endif

#ifdef TVM_VULKAN_RUNTIME
#include "../src/runtime/vulkan/vulkan.cc"
#endif

#ifdef USE_SORT
#include "../src/runtime/contrib/sort/sort.cc"
#endif

#ifdef USE_RANDOM
#include "../src/runtime/contrib/random/random.cc"
#endif

#include <android/log.h>

namespace tvm {
namespace runtime{
namespace detail{
// Override logging mechanism
class LogFatal {
 public:
  LogFatal(const std::string& file, int lineno) : file_(file), lineno_(lineno) {}
  ~LogFatal() TVM_THROW_EXCEPTION {
    __android_log_write(ANDROID_LOG_DEBUG, "TVM_RUNTIME", stream_.str().c_str());
    throw InternalError(file_, lineno_, stream_.str()); }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  std::string file_;
  int lineno_;
};

class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno) {
    std::time_t t = std::time(nullptr);
    stream_ << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " << file << ":" << lineno
            << ": ";
  }
  ~LogMessage() {

  __android_log_write(ANDROID_LOG_DEBUG, "TVM_RUNTIME", stream_.str().c_str());
  }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
};
}
}
}  // namespace dmlc
