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
 * \file rpc_env.cc
 * \brief Server environment of the RPC.
 */
#include "../cpp_rpc/rpc_env.h"

#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <dirent.h>
#include <remote.h>
#include <rpcmem.h>
#include <sys/stat.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../../src/support/utils.h"
#include "common.h"
#include "hexagon_proxy_rpc.h"

namespace tvm {
namespace runtime {

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname THe name of the directory
 */
void CleanDir(const std::string& dirname);

namespace hexagon {
using FastRPCHandle = remote_handle64;
using Handle = uint32_t;

AEEResult enable_unsigned_pd(bool enable) {
  remote_rpc_control_unsigned_module data;
  data.domain = CDSP_DOMAIN_ID;
  data.enable = static_cast<int>(enable);
  AEEResult rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
  if (rc != AEE_SUCCESS) {
    std::cout << "error " << (enable ? "enabling" : "disabling") << " unsigned PD\n";
  }
  return rc;
}

AEEResult set_remote_stack_size(int size) {
  remote_rpc_thread_params data;
  data.domain = CDSP_DOMAIN_ID;
  data.prio = -1;
  data.stack_size = size;
  AEEResult rc = remote_session_control(FASTRPC_THREAD_PARAMS, &data, sizeof(data));
  if (rc != AEE_SUCCESS) {
    std::cout << "error setting remote stack size: " << std::hex << rc << '\n';
  }
  return rc;
}

class FastRPCChannel {
 public:
  explicit FastRPCChannel(const std::string& uri) {
    enable_unsigned_pd(true);
    set_remote_stack_size(128 * 1024);

    int rc = hexagon_proxy_rpc_open(uri.c_str(), &handle_);
    if (rc != AEE_SUCCESS) {
      handle_ = std::numeric_limits<uint64_t>::max();
    }
  }

  ~FastRPCChannel() {
    if (handle_ == std::numeric_limits<uint64_t>::max()) {
      return;
    }

    hexagon_proxy_rpc_close(handle_);
    handle_ = std::numeric_limits<uint64_t>::max();
  }

  FastRPCHandle GetHandle() { return handle_; }

 private:
  FastRPCHandle handle_ = std::numeric_limits<uint64_t>::max();
};

class HexagonModuleNode : public ModuleNode {
 public:
  HexagonModuleNode() = delete;
  HexagonModuleNode(FastRPCHandle h, std::string file_name) : handle_(h), mod_{0} {
    AEEResult rc = hexagon_proxy_rpc_load(handle_, file_name.c_str(), &mod_);
    if (rc != AEE_SUCCESS) {
      LOG(FATAL) << "Error loading module\n";
    }
  }
  ~HexagonModuleNode() {
    AEEResult rc = hexagon_proxy_rpc_unload(handle_, mod_);
    if (rc != AEE_SUCCESS) {
      LOG(FATAL) << "Error unloading module\n";
    }
    for (Handle func : packed_func_handles_) {
      AEEResult rc = hexagon_proxy_rpc_release_function(handle_, func);
      if (rc != AEE_SUCCESS) {
        LOG(FATAL) << "Error releasing function\n";
      }
    }
  }
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    hexagon::Handle func;
    AEEResult rc = hexagon_proxy_rpc_get_function(handle_, name.c_str(), mod_, &func);
    if (rc != AEE_SUCCESS) {
      LOG(FATAL) << "Error calling get_function\n";
    }
    packed_func_handles_.push_back(func);
    return PackedFunc([handle = this->handle_, func, name](TVMArgs args, TVMRetValue* rv) {
      std::vector<uint32_t> handles;
      for (size_t i = 0; i < args.size(); i++) {
        ICHECK_EQ(args.type_codes[i], kTVMDLTensorHandle);
        DLTensor* tensor = args[i];
        auto f = runtime::Registry::Get("runtime.hexagon.GetHandle");
        int32_t thandle = (*f)(tensor->data);
        handles.push_back(thandle);
      }
      auto* packet = reinterpret_cast<HandlePacket*>(rpcmem_alloc(
          RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, HandlePacket::size(args.size())));
      packet->ndim = args.size();
      std::copy(handles.begin(), handles.end(), packet->handles);
      AEEResult rc = hexagon_proxy_rpc_invoke(
          handle, func, reinterpret_cast<const unsigned char*>(packet), packet->size());
      if (rc != AEE_SUCCESS) {
        LOG(FATAL) << "Error invoking function: " << name;
      }
      rpcmem_free(packet);
    });
  }
  const char* type_key() const { return "HexagonModule"; }

 private:
  FastRPCHandle handle_;
  Handle mod_;
  std::vector<Handle> packed_func_handles_;
};
}  // namespace hexagon

RPCEnv::RPCEnv(const std::string& wd) {
  if (wd != "") {
    base_ = wd + "/.cache";
    mkdir(wd.c_str(), 0777);
    mkdir(base_.c_str(), 0777);
  } else {
    char cwd[PATH_MAX];
    auto cmdline = fopen("/proc/self/cmdline", "r");
    fread(cwd, 1, sizeof(cwd), cmdline);
    fclose(cmdline);
    std::string android_base_ = "/data/data/" + std::string(cwd) + "/cache";
    struct stat statbuf;
    // Check if application data directory exist. If not exist, usually means we run tvm_rpc from
    // adb shell terminal.
    if (stat(android_base_.data(), &statbuf) == -1 || !S_ISDIR(statbuf.st_mode)) {
      // Tmp directory is always writable for 'shell' user.
      android_base_ = "/data/local/tmp";
    }
    base_ = android_base_ + "/rpc";
    mkdir(base_.c_str(), 0777);
  }

  static hexagon::FastRPCChannel hexagon_proxy_rpc(hexagon_proxy_rpc_URI CDSP_DOMAIN);
  if (hexagon_proxy_rpc.GetHandle() == -1) {
    LOG(FATAL) << "Error opening FastRPC channel\n";
  }

  TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath").set_body([this](TVMArgs args, TVMRetValue* rv) {
    *rv = this->GetPath(args[0]);
  });

  TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module")
      .set_body([this, handle = hexagon_proxy_rpc.GetHandle()](TVMArgs args, TVMRetValue* rv) {
        std::string file_name = this->GetPath(args[0]);
        auto n = make_object<hexagon::HexagonModuleNode>(handle, file_name);
        *rv = Module(n);
        LOG(INFO) << "Load module from " << file_name << " ...";
      });

  TVM_REGISTER_GLOBAL("tvm.rpc.hexagon.allocate")
      .set_body([handle = hexagon_proxy_rpc.GetHandle()](TVMArgs args, TVMRetValue* rv) {
        DLTensor* ext_tensor = args[0];
        Optional<String> mem_scope = args[1];

        auto* input_meta = reinterpret_cast<tensor_meta*>(rpcmem_alloc(
            RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, tensor_meta::meta_size(ext_tensor->ndim)));
        input_meta->ndim = ext_tensor->ndim;
        input_meta->dtype = ext_tensor->dtype;
        std::copy(ext_tensor->shape, ext_tensor->shape + ext_tensor->ndim, input_meta->shape);

        hexagon::Handle hexagon_buffer;
        const char* scope = mem_scope.defined() ? mem_scope.value().c_str() : "";
        AEEResult rc =
            hexagon_proxy_rpc_allocate(handle, reinterpret_cast<const unsigned char*>(input_meta),
                                       input_meta->meta_size(), scope, &hexagon_buffer);
        if (rc != AEE_SUCCESS) {
          LOG(FATAL) << "Error allocating hexagon ndrray\n";
        }
        rpcmem_free(input_meta);
        *rv = static_cast<int32_t>(hexagon_buffer);
        return rc == AEE_SUCCESS;
      });

  TVM_REGISTER_GLOBAL("tvm.rpc.hexagon.read_to_host")
      .set_body([handle = hexagon_proxy_rpc.GetHandle()](TVMArgs args, TVMRetValue* rv) {
        void* host_ptr = static_cast<void*>(args[0]);
        size_t nbytes = args[1];
        hexagon::Handle hexagon_buffer = static_cast<int32_t>(args[2]);
        AEEResult rc = hexagon_proxy_rpc_read(handle, static_cast<uint8_t*>(host_ptr),
                                              static_cast<int32_t>(nbytes), hexagon_buffer);
        if (rc != AEE_SUCCESS) {
          LOG(FATAL) << "Error reading from hexagon buffer\n";
        }
      });

  TVM_REGISTER_GLOBAL("tvm.rpc.hexagon.write_from_host")
      .set_body([handle = hexagon_proxy_rpc.GetHandle()](TVMArgs args, TVMRetValue* rv) {
        hexagon::Handle hexagon_buffer = static_cast<int32_t>(args[0]);
        void* host_ptr = static_cast<void*>(args[1]);
        size_t nbytes = args[2];
        AEEResult rc = hexagon_proxy_rpc_write(
            handle, hexagon_buffer, static_cast<uint8_t*>(host_ptr), static_cast<int32_t>(nbytes));
        if (rc != AEE_SUCCESS) {
          LOG(FATAL) << "Error writing to hexagon buffer\n";
        }
      });

  TVM_REGISTER_GLOBAL("tvm.rpc.hexagon.release")
      .set_body([handle = hexagon_proxy_rpc.GetHandle()](TVMArgs args, TVMRetValue* rv) {
        hexagon::Handle hexagon_buffer = static_cast<int32_t>(args[0]);
        AEEResult rc = hexagon_proxy_rpc_release(handle, hexagon_buffer);
        if (rc != AEE_SUCCESS) {
          LOG(FATAL) << "Error writing to hexagon buffer\n";
        }
      });
}

/*!
 * \brief GetPath To get the work path from packed function
 * \param file_name The file name
 * \return The full path of file.
 */
std::string RPCEnv::GetPath(const std::string& file_name) const {
  // we assume file_name has "/" means file_name is the exact path
  // and does not create /.rpc/
  return file_name.find('/') != std::string::npos ? file_name : base_ + "/" + file_name;
}
/*!
 * \brief Remove The RPC Environment cleanup function
 */
void RPCEnv::CleanUp() const {
  CleanDir(base_);
  const int ret = rmdir(base_.c_str());
  if (ret != 0) {
    LOG(WARNING) << "Remove directory " << base_ << " failed";
  }
}

/*!
 * \brief ListDir get the list of files in a directory
 * \param dirname The root directory name
 * \return vector Files in directory.
 */
std::vector<std::string> ListDir(const std::string& dirname) {
  std::vector<std::string> vec;
  DIR* dp = opendir(dirname.c_str());
  if (dp == nullptr) {
    int errsv = errno;
    LOG(FATAL) << "ListDir " << dirname << " error: " << strerror(errsv);
  }
  dirent* d;
  while ((d = readdir(dp)) != nullptr) {
    std::string filename = d->d_name;
    if (filename != "." && filename != "..") {
      std::string f = dirname;
      if (f[f.length() - 1] != '/') {
        f += '/';
      }
      f += d->d_name;
      vec.push_back(f);
    }
  }
  closedir(dp);
  return vec;
}

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname The name of the directory
 */
void CleanDir(const std::string& dirname) {
  auto files = ListDir(dirname);
  for (const auto& filename : files) {
    std::string file_path = dirname + "/";
    file_path += filename;
    const int ret = std::remove(filename.c_str());
    if (ret != 0) {
      LOG(WARNING) << "Remove file " << filename << " failed";
    }
  }
}
}  // namespace runtime
}  // namespace tvm
