/*!
 *  Copyright (c) 2018 by Contributors
 * \file cur_module.cc
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
  thread_local SGXModuleNode* cur_mod;
}  // namespace sgx

class SGXModuleNode : public ModuleNode {
 public:
  ~SGXModuleNode() {
    if (eid_) sgx_destroy_enclave(eid_);
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

    sgx::cur_mod = this;
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
          sgx::cur_mod = this;
          sgx_status = tvm_ecall_packed_func(eid_, ecall_name.c_str(), &args, rv);
          CHECK_EQ(sgx_status, SGX_SUCCESS) << "SGX Error: " << sgx_status;
        });
    }
    return PackedFunc();
  }

  void RegisterFunc(const std::string& name) {
    exports_.insert(name);
  }

  void RunWorker(int num_tasks, const void* cb) {
    std::function<void(int)> runner = [this, cb](int _worker_id) {
      sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
      sgx_status = tvm_ecall_run_worker(this->eid_, cb);
      CHECK(sgx_status == SGX_SUCCESS) << "SGX Error: " << sgx_status;
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

void tvm_ocall_thread_group_launch(int num_tasks, const void* cb) {
  cur_mod->RunWorker(num_tasks, cb);
}

void tvm_ocall_thread_group_join() {
  cur_mod->JoinThreads();
}

void tvm_ocall_api_set_last_error(const char* err) {
  TVMAPISetLastError(err);
}

void tvm_ocall_register_func(const char* name) {
  cur_mod->RegisterFunc(name);
}

}  // extern "C"
}  // namespace sgx

}  // namespace runtime
}  // namespace tvm
