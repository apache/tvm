#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "../../src/runtime/sgx/trusted/runtime.h"
#include "../../src/runtime/sgx/trusted/runtime.cc"

using namespace tvm::runtime;

extern "C" {
void tvm_ecall_init() {}

void tvm_ecall_packed_func(const char* cname,
                           void* tvm_args,
                           void* tvm_ret_val) {
  std::string name = std::string(cname);
  CHECK(name.substr(0, sgx::ECALL_PACKED_PFX.size()) == sgx::ECALL_PACKED_PFX)
    << "Function `" << name << "` is not an enclave export.";
  const PackedFunc* f = Registry::Get(name);
  CHECK(f != nullptr) << "Enclave function not found.";
  f->CallPacked(*reinterpret_cast<TVMArgs*>(tvm_args),
      reinterpret_cast<TVMRetValue*>(tvm_ret_val));
}

}

TVM_REGISTER_ENCLAVE_FUNC("__tvm_main__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module mod = (*Registry::Get("module._GetSystemLib"))();
    mod.GetFunction("addonesys").CallPacked(args, rv);
  });
