#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>

void test_double() {
  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  tvm::runtime::PackedFunc f = mod.GetFunction("tvmgen_default_fused_add");

  DLTensor* x;
  DLTensor* y;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);
  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(x->data)[i] = i;
  }
  f(x, y);
  for (int i = 0; i < shape[0]; ++i) {
    LOG(INFO) << static_cast<float*>(x->data)[i] << "*2=" << static_cast<float*>(y->data)[i];
  }
  TVMArrayFree(x);
  TVMArrayFree(y);
}

int main(void) {
  test_double();
  return 0;
}
