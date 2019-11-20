#include <tvm/runtime/c_runtime_api.h>
#include <cstdint>

TVM_DLL extern "C" float Nop32ToFloat(uint32_t in) {
  return 1.0;
}

TVM_DLL extern "C" uint32_t FloatToNop32(float in) {
  return 1;
}

// TODO(gus) how wide should the input be?
TVM_DLL extern "C" uint32_t IntToNop32(int in) {
  return in;
}

TVM_DLL extern "C" uint32_t Nop32Add(uint32_t a, uint32_t b) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Sub(uint32_t a, uint32_t b) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Mul(uint32_t a, uint32_t b) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Div(uint32_t a, uint32_t b) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Max(uint32_t a, uint32_t b) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Sqrt(uint32_t a) {
  return a;
}

TVM_DLL extern "C" uint32_t Nop32Exp(uint32_t a) {
  return a;
}
