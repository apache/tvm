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

#import <Foundation/Foundation.h>

#include <tvm/runtime/registry.h>

#include "RPCArgs.h"

// internal TVM header
#include <../../../src/runtime/file_utils.h>

#if defined(USE_CUSTOM_DSO_LOADER) && USE_CUSTOM_DSO_LOADER == 1
// internal TVM header to achieve Library class
#include <../../../src/runtime/library_module.h>
#include <custom_dlfcn.h>
#endif

namespace tvm {
namespace runtime {
namespace detail {

// Override logging mechanism
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  throw tvm::runtime::InternalError(file, lineno, message);
}

void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  NSLog(@"%s:%d: %s", file.c_str(), lineno, message.c_str());
}

}  // namespace detail

TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([](TVMArgs args, TVMRetValue* rv) {
  static const std::string base_ = NSTemporaryDirectory().UTF8String;
  const std::string path = args[0];
  *rv = base_ + "/" + path;
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
