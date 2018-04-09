/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx_module.cc
 * \brief SGX enclave module.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>
#include <sgx_urts.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include "../common.h"
#include "../../file_util.h"

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
    TVM_SGX_CHECKED_CALL(tvm_ecall_packed_func(eid_, 0, nullptr, nullptr, 0, &rv));
    std::string exports = rv;
    std::istringstream exports_iss(exports);
    std::copy(std::istream_iterator<std::string>(exports_iss),
        std::istream_iterator<std::string>(),
        std::back_inserter(exports_));

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
    size_t func_id = std::distance(exports_.begin(),
        std::find(exports_.begin(), exports_.end(), name));
    CHECK_LT(func_id, exports_.size())
      << "\"" << name << "\" is not an enclave export.";
    if (func_id >= exports_.size()) return PackedFunc();
    return PackedFunc([this, func_id](TVMArgs args, TVMRetValue* rv) {
        sgx::EnclaveContext ctx(this);
        TVM_SGX_CHECKED_CALL(tvm_ecall_packed_func(eid_, func_id,
              args.values, args.type_codes, args.num_args, rv));
      });
  }

  void RunWorkers(int num_tasks, void* tg) {
    std::function<void(int)> runner = [this, tg](int _worker_id) {
      this->GetFunction("__tvm_run_worker__",
                        std::shared_ptr<SGXModuleNode>(nullptr))(tg);
    };
    thread_group_.reset(new tvm::runtime::threading::ThreadGroup(
          num_tasks, runner, false /* include_main_thread */));
  }

  void JoinThreads() {
    thread_group_->Join();
  }

 private:
  // ID of the loaded enclave
  sgx_enclave_id_t eid_;
  // Names and IDs of functions exported by the enclave module
  std::vector<std::string> exports_;
  std::unique_ptr<tvm::runtime::threading::ThreadGroup> thread_group_;
};

namespace sgx {

TVM_REGISTER_GLOBAL("__sgx_thread_group_launch__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  EnclaveContext::GetModule()->RunWorkers(args[0], args[1]);
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

extern "C" {

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
  rv.MoveToCHost(ret_val, ret_type_code);  // only support POD types for now
}

void* tvm_ocall_malloc(size_t num_bytes) {
  void* buf = calloc(1, num_bytes);
  CHECK(buf != nullptr);
  return buf;
}

void tvm_ocall_free(void* ptr) {
  free(ptr);
}

void tvm_ocall_set_return(TVMRetValueHandle ret,
                           const TVMValue* value,
                           const int* type_code,
                           int num_ret) {
  CHECK_EQ(num_ret, 1) << "Only one return value is currently supported.";
  CHECK(type_code[0] != kStr) << "Return kBytes, not kStr.";
  TVMRetValue* rv = static_cast<TVMRetValue*>(ret);
  *rv = TVMArgValue(value[0], type_code[0]);
  if (type_code[0] == kBytes) free(value[0].v_handle);
}

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
