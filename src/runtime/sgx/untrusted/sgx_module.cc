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
 *  Copyright (c) 2018 by Contributors
 * \file sgx_module.cc
 * \brief SGX enclave module.
 */
#include <dmlc/logging.h>
#include <sgx_urts.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include "../common.h"
#include "../../file_util.h"
#include "./tvm_u.h"

namespace tvm {
namespace runtime {

class SGXModuleNode;

namespace sgx {

class EnclaveContext {
 public:
  explicit EnclaveContext(SGXModuleNode* mod) {
    CHECK(Context()->mod_ == nullptr)
      << "Tried overriding existing enclave context.";
    CHECK(mod != nullptr) << "Tried setting null enclave context.";
    Context()->mod_ = mod;
  }
  ~EnclaveContext() {
    Context()->mod_ = nullptr;
  }

  static SGXModuleNode* GetModule() {
    SGXModuleNode* ctx = Context()->mod_;
    CHECK(ctx != nullptr) << "No current enclave context";
    return ctx;
  }

 private:
  EnclaveContext() {}
  SGXModuleNode* mod_;

  static EnclaveContext* Context() {
    static thread_local EnclaveContext inst;
    return &inst;
  }
};

}  // namespace sgx

class SGXModuleNode : public ModuleNode {
 public:
  ~SGXModuleNode() {
    if (eid_) {
      sgx::EnclaveContext ctx(this);
      sgx_destroy_enclave(eid_);
    }
  }

  void Init(const std::string& enclave_file) {
    std::string token_file = GetCacheDir() + "/" +
                             GetFileBasename(enclave_file) + ".token";
    sgx_launch_token_t token = {0};
    int token_updated = 0;

    try {
      std::ifstream ifs(token_file, std::fstream::in | std::fstream::binary);
      ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      ifs >> token;
    } catch (std::ifstream::failure e) {
      memset(&token, 0x0, sizeof(sgx_launch_token_t));
    }

    TVM_SGX_CHECKED_CALL(sgx_create_enclave(
        enclave_file.c_str(), SGX_DEBUG_FLAG, &token, &token_updated, &eid_, NULL));

    sgx::EnclaveContext ctx(this);
    TVMRetValue rv;
    TVM_SGX_CHECKED_CALL(tvm_ecall_init(eid_, &rv));

    if (!token_updated) return;

    try {
      std::ofstream ofs(token_file, std::fstream::trunc | std::fstream::binary);
      ofs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      ofs << token;
    } catch (std::ifstream::failure e) {
      LOG(INFO) << "Could not save SGX launch token to " << token_file;
    }
  }

  const char* type_key() const final {
    return "sgx";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    auto exported = exports_.find(name);
    if (exported == exports_.end()) return PackedFunc();
    int func_id = exported->second;
    return PackedFunc([this, func_id](TVMArgs args, TVMRetValue* rv) {
        sgx::EnclaveContext ctx(this);
        TVMValue ret_value;
        int ret_type_code;
        TVM_SGX_CHECKED_CALL(tvm_ecall_packed_func(eid_, func_id,
              args.values, args.type_codes, args.num_args, &ret_value, &ret_type_code));
        *rv = TVMArgValue(ret_value, ret_type_code);
      });
  }

  void RunWorkers(int num_tasks) {
    std::function<void(int)> runner = [this](int _worker_id) {
      this->GetFunction("__tvm_run_worker__",
                        std::shared_ptr<SGXModuleNode>(nullptr))();
    };
    thread_group_.reset(new tvm::runtime::threading::ThreadGroup(
          num_tasks, runner, false /* include_main_thread */));
  }

  void JoinThreads() {
    thread_group_->Join();
  }

  void RegisterExport(std::string name, int func_id) {
    exports_[name] = func_id;
  }

 private:
  // ID of the loaded enclave
  sgx_enclave_id_t eid_;
  // Names and IDs of functions exported by the enclave module
  std::unordered_map<std::string, int> exports_;
  std::unique_ptr<tvm::runtime::threading::ThreadGroup> thread_group_;
};

namespace sgx {

TVM_REGISTER_GLOBAL("__sgx_thread_group_launch__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  EnclaveContext::GetModule()->RunWorkers(args[0]);
});

TVM_REGISTER_GLOBAL("__sgx_thread_group_join__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  EnclaveContext::GetModule()->JoinThreads();
});

TVM_REGISTER_GLOBAL("__sgx_set_last_error__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string err = args[0];
  TVMAPISetLastError(err.c_str());
});

TVM_REGISTER_GLOBAL("__sgx_println__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::ostringstream msg;
  for (int i = 0; i < args.num_args; ++i) {
    switch (args.type_codes[i]) {
    case kDLInt: msg << static_cast<int64_t>(args[i]); break;
    case kDLUInt: msg << static_cast<uint64_t>(args[i]); break;
    case kDLFloat: msg << static_cast<double>(args[i]); break;
    case kStr:
    case kBytes: {
      std::string val = args[i];
      msg << val;
    }
    break;
    }
    msg << " ";
  }
  LOG(INFO) << msg.str();
});

extern "C" {

void tvm_ocall_register_export(const char* name, int func_id) {
  EnclaveContext::GetModule()->RegisterExport(name, func_id);
}

void tvm_ocall_packed_func(const char* name,
                           const TVMValue* arg_values,
                           const int* type_codes,
                           int num_args,
                           TVMValue* ret_val,
                           int* ret_type_code) {
  const PackedFunc* f = Registry::Get(name);
  CHECK(f != nullptr) << "ocall to nonexistent function \"" << name << "\"";
  TVMRetValue rv;
  f->CallPacked(TVMArgs(arg_values, type_codes, num_args), &rv);
  rv.MoveToCHost(ret_val, ret_type_code);
}

// Allocates space for return values. The returned pointer is only valid between
// successive calls to `tvm_ocall_reserve_space`.
TVM_REGISTER_GLOBAL("__sgx_reserve_space__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  size_t num_bytes = args[0];
  size_t alignment = args[1];

  static TVMContext ctx = { kDLCPU, 0 };
  static thread_local void* buf = nullptr;
  static thread_local size_t buf_size = 0;
  static thread_local size_t buf_align = 0;

  if (buf_size >= num_bytes && buf_align >= alignment) *rv = nullptr;

  DeviceAPI::Get(ctx)->FreeDataSpace(ctx, buf);
  buf = DeviceAPI::Get(ctx)->AllocDataSpace(ctx, num_bytes, alignment, {});
  buf_size = num_bytes;
  buf_align = alignment;

  *rv = buf;
});

}  // extern "C"
}  // namespace sgx

TVM_REGISTER_GLOBAL("module.loadfile_sgx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::shared_ptr<SGXModuleNode> node = std::make_shared<SGXModuleNode>();
  node->Init(args[0]);
  *rv = runtime::Module(node);
});

}  // namespace runtime
}  // namespace tvm
