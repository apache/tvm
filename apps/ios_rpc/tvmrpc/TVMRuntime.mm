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
 * \file TVMRuntime.mm
 */
#include "TVMRuntime.h"
// Runtime API
#include "../../../src/runtime/c_runtime_api.cc"
#include "../../../src/runtime/cpu_device_api.cc"
#include "../../../src/runtime/dso_library.cc"
#include "../../../src/runtime/file_utils.cc"
#include "../../../src/runtime/library_module.cc"
#include "../../../src/runtime/logging.cc"
#include "../../../src/runtime/metadata_module.cc"
#include "../../../src/runtime/module.cc"
#include "../../../src/runtime/ndarray.cc"
#include "../../../src/runtime/object.cc"
#include "../../../src/runtime/profiling.cc"
#include "../../../src/runtime/registry.cc"
#include "../../../src/runtime/source_utils.cc"
#include "../../../src/runtime/system_library.cc"
#include "../../../src/runtime/thread_pool.cc"
#include "../../../src/runtime/threading_backend.cc"
#include "../../../src/runtime/workspace_pool.cc"

// RPC server
#include "../../../src/runtime/rpc/rpc_channel.cc"
#include "../../../src/runtime/rpc/rpc_endpoint.cc"
#include "../../../src/runtime/rpc/rpc_local_session.cc"
#include "../../../src/runtime/rpc/rpc_module.cc"
#include "../../../src/runtime/rpc/rpc_server_env.cc"
#include "../../../src/runtime/rpc/rpc_session.cc"
#include "../../../src/runtime/rpc/rpc_socket_impl.cc"
// Graph executor
#include "../../../src/runtime/graph_executor/graph_executor.cc"
// Metal
#include "../../../src/runtime/metal/metal_device_api.mm"
#include "../../../src/runtime/metal/metal_module.mm"
// CoreML
#include "../../../src/runtime/contrib/coreml/coreml_runtime.mm"

#if defined(USE_CUSTOM_DSO_LOADER) && USE_CUSTOM_DSO_LOADER == 1
#include <custom_dlfcn.h>
#endif

namespace tvm {
namespace runtime {
namespace detail {
// Override logging mechanism
void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  throw tvm::runtime::InternalError(file, lineno, message);
}

void LogMessageImpl(const std::string& file, int lineno, const std::string& message) {
  NSLog(@"%s:%d: %s", file.c_str(), lineno, message.c_str());
}
}
}
}  // namespace dmlc

namespace tvm {
namespace runtime {

class NSStreamChannel final : public RPCChannel {
 public:
  explicit NSStreamChannel(NSOutputStream* stream) : stream_(stream) {}

  size_t Send(const void* data, size_t size) final {
    ssize_t nbytes = [stream_ write:reinterpret_cast<const uint8_t*>(data) maxLength:size];
    if (nbytes < 0) {
      NSLog(@"%@", [stream_ streamError].localizedDescription);
      throw tvm::Error("Stream error");
    }
    return nbytes;
  }

  size_t Recv(void* data, size_t size) final {
    LOG(FATAL) << "Do not allow explicit receive for";
    return 0;
  }

 private:
  NSOutputStream* stream_;
};

FEventHandler CreateServerEventHandler(NSOutputStream* outputStream, std::string name,
                                       std::string remote_key) {
  std::unique_ptr<NSStreamChannel> ch(new NSStreamChannel(outputStream));
  std::shared_ptr<RPCEndpoint> sess = RPCEndpoint::Create(std::move(ch), name, remote_key);
  return [sess](const std::string& in_bytes, int flag) {
    return sess->ServerAsyncIOEventHandler(in_bytes, flag);
  };
}

// Runtime environment
struct RPCEnv {
 public:
  RPCEnv() {
    NSString* path = NSTemporaryDirectory();
    base_ = [path UTF8String];
    if (base_[base_.length() - 1] != '/') {
      base_ = base_ + '/';
    }
  }
  // Get Path.
  std::string GetPath(const std::string& file_name) { return base_ + file_name; }

 private:
  std::string base_;
};

void LaunchSyncServer() {
  // only load dylib from frameworks.
  NSBundle* bundle = [NSBundle mainBundle];
  NSString* base = [bundle privateFrameworksPath];
  NSString* path = [base stringByAppendingPathComponent:@"tvm/rpc_config.txt"];
  std::string name = [path UTF8String];
  std::ifstream fs(name, std::ios::in);
  std::string url, key;
  int port;
  ICHECK(fs >> url >> port >> key) << "Invalid RPC config file " << name;
  RPCConnect(url, port, "server:" + key, TVMArgs(nullptr, nullptr, 0))->ServerLoop();
}

TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([](TVMArgs args, TVMRetValue* rv) {
  static RPCEnv env;
  *rv = env.GetPath(args[0]);
});

TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string name = args[0];
  std::string fmt = GetFileFormat(name, "");
  NSString* base;
  if (fmt == "dylib") {
    // only load dylib from frameworks.
    NSBundle* bundle = [NSBundle mainBundle];
    base = [[bundle privateFrameworksPath] stringByAppendingPathComponent:@"tvm"];

    if (Registry::Get("runtime.module.loadfile_dylib_custom")) {
      // Custom dso laoder is present. Will use it.
      base = NSTemporaryDirectory();
      fmt = "dylib_custom";
    }
  } else {
    // Load other modules in tempdir.
    base = NSTemporaryDirectory();
  }
  NSString* path =
      [base stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
  name = [path UTF8String];
  *rv = Module::LoadFromFile(name, fmt);
  LOG(INFO) << "Load module from " << name << " ...";
});

#if defined(USE_CUSTOM_DSO_LOADER) && USE_CUSTOM_DSO_LOADER == 1

// Custom dynamic library loader. Supports unsigned binary
class UnsignedDSOLoader final : public Library {
 public:
  ~UnsignedDSOLoader() {
    if (lib_handle_) {
      custom_dlclose(lib_handle_);
      lib_handle_ = nullptr;
    };
  }
  void Init(const std::string& name) {
    lib_handle_ = custom_dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    ICHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name << " " << custom_dlerror();
  }

  void* GetSymbol(const char* name) final { return custom_dlsym(lib_handle_, name); }

 private:
  // Library handle
  void* lib_handle_{nullptr};
};

// Add UnsignedDSOLoader plugin in global registry
TVM_REGISTER_GLOBAL("runtime.module.loadfile_dylib_custom")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      auto n = make_object<UnsignedDSOLoader>();
      n->Init(args[0]);
      *rv = CreateModuleFromLibrary(n);
    });

#endif

}  // namespace runtime
}  // namespace tvm

@implementation TVMRuntime

+ (void)launchSyncServer {
  tvm::runtime::LaunchSyncServer();
}

@end
