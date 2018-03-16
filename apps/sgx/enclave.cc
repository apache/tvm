#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#ifndef _LIBCPP_SGX_CONFIG
#include <iostream>
#endif

/* This function mirrors the one in howto_deploy except without the iostream */
int Verify(tvm::runtime::Module mod, std::string fname) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);

  // Allocate the DLPack data structures.
  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &y);
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }

  // Invoke the function
  f(x, y);

  // check the output
  bool all_eq = true;
  for (int i = 0; i < shape[0]; ++i) {
    all_eq = all_eq && static_cast<float*>(y->data)[i] == i + 1.0f;
  }

  return all_eq;
}


extern "C" {
void tvm_ecall_run_module(const void* tvm_args, void* tvm_return_value) {
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("module._GetSystemLib"))();
  *(int*)tvm_return_value = Verify(mod_syslib, "addonesys");
}
}

#ifndef _LIBCPP_SGX_CONFIG
int main(void) {
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("module._GetSystemLib"))();
  if (Verify(mod_syslib, "addonesys")) {
    std::cout << "It works!" << std::endl;
    return 0;
  }
  std::cerr << "It doesn't work." << std::endl;
  return -1;
}
#endif
