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
#include <fstream>
#include "../common.h"
#include "../../file_util.h"

namespace tvm {
namespace runtime {

class SGXModuleNode;

namespace sgx {

class EnclaveContext {
 public:
  explicit EnclaveContext(SGXModuleNode* mod) {
    CHECK(Global()->mod_ == nullptr)
      << "Tried overriding existing enclave context.";
    CHECK(mod != nullptr) << "Tried setting null enclave context.";
    Global()->mod_ = mod;
  }
  ~EnclaveContext() {
    Global()->mod_ = nullptr;
  }

  static EnclaveContext* Global() {
    static EnclaveContext inst;
    return &inst;
  }

  static SGXModuleNode* GetCurrent() {
    SGXModuleNode* ctx = Global()->mod_;
    CHECK(ctx != nullptr) << "No current enclave context";
    return ctx;
  }

 private:
  EnclaveContext() {}
  SGXModuleNode* mod_;
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
    sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
    int token_updated = 0;

    try {
      std::ifstream ifs(token_file, std::fstream::in | std::fstream::binary);
      ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      ifs >> token;
    } catch (std::ifstream::failure e) {
      memset(&token, 0x0, sizeof(sgx_launch_token_t));
    }

    sgx_status = sgx_create_enclave(
        enclave_file.c_str(), SGX_DEBUG_FLAG, &token, &token_updated, &eid_, NULL);
    CHECK_EQ(sgx_status, SGX_SUCCESS)
      << "Failed to load enclave. SGX Error: " << sgx_status;

    sgx::EnclaveContext ctx(this);
    sgx_status = tvm_ecall_init(eid_);
    CHECK_EQ(sgx_status, SGX_SUCCESS)
      << "Failed to initialize enclave. SGX Error: " << sgx_status;

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
    std::string ecall_name = sgx::ECALL_PACKED_PFX + name;
    auto it = exports_.find(name);
    if (it != exports_.end()) {
      return PackedFunc([this, ecall_name](TVMArgs args, TVMRetValue* rv) {
          sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
          sgx::EnclaveContext ctx(this);
          sgx_status = tvm_ecall_packed_func(eid_,
                                             ecall_name.c_str(),
                                             args.values,
                                             args.type_codes,
                                             args.num_args,
                                             rv);
          CHECK_EQ(sgx_status, SGX_SUCCESS) << "SGX Error: " << sgx_status;
        });
    }
    return PackedFunc();
  }

  void RegisterFunc(const std::string& name) {
    exports_.insert(name);
  }

  void RunWorkers(int num_tasks, void* tg) {
    std::function<void(int)> runner = [this, tg](int _worker_id) {
      this->GetFunction("__tvm_run_worker__",
                        std::shared_ptr<SGXModuleNode>(this))(tg);
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
  std::unordered_set<std::string> exports_;
  std::unique_ptr<tvm::runtime::threading::ThreadGroup> thread_group_;
};

TVM_REGISTER_GLOBAL("module.loadfile_sgx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<SGXModuleNode> node = std::make_shared<SGXModuleNode>();
    node->Init(args[0]);
    *rv = runtime::Module(node);
  });

namespace sgx {
extern "C" {

void tvm_ocall_thread_group_launch(int num_tasks, void* tg) {
  EnclaveContext::GetCurrent()->RunWorkers(num_tasks, tg);
}

void tvm_ocall_thread_group_join() {
  EnclaveContext::GetCurrent()->JoinThreads();
}

void tvm_ocall_api_set_last_error(const char* err) {
  TVMAPISetLastError(err);
}

void tvm_ocall_register_func(const char* name) {
  EnclaveContext::GetCurrent()->RegisterFunc(name);
}

}  // extern "C"
}  // namespace sgx

}  // namespace runtime
}  // namespace tvm
