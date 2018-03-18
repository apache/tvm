/*!
 *  Copyright (c) 2018 by Contributors
 * \file sgx_module.cc
 * \brief SGX enclave module.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <sgx_eid.h>
#include <sgx_urts.h>
#include <cstring>
#include <fstream>
#include "../../file_util.h"
#include "../../module_util.h"
#include "runtime.h"

namespace tvm {
namespace runtime {

namespace sgx {
  thread_local sgx_enclave_id_t last_eid;
}  // sgx

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
    if (name == symbol::tvm_module_main || name == "ecall_tvm_main") {
      return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
          sgx_status_t sgx_status = SGX_ERROR_UNEXPECTED;
          sgx::last_eid = eid_;
          sgx_status = ecall_tvm_main(eid_,
              const_cast<TVMValue*>(args.values),
              const_cast<int*>(args.type_codes),
              args.num_args);
          CHECK_EQ(sgx_status, SGX_SUCCESS) << "SGX Error: " << sgx_status;
        });
    }
    return PackedFunc();
  }

 private:
  sgx_enclave_id_t eid_;
};

TVM_REGISTER_GLOBAL("module.loadfile_sgx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<SGXModuleNode> node = std::make_shared<SGXModuleNode>();
    node->Init(args[0]);
    *rv = runtime::Module(node);
  });

}  // namespace runtime
}  // namespace tvm
