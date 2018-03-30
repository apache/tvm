#include "../../src/runtime/sgx/trusted/runtime.cc"

using namespace tvm::runtime;

TVM_REGISTER_ENCLAVE_FUNC("__tvm_main__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module mod = (*Registry::Get("module._GetSystemLib"))();
    mod.GetFunction("addonesys").CallPacked(args, rv);
  });
