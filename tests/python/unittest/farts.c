#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/runtime/utvm_device_lib.h"
extern void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_add( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 2))) {
    TVMAPISetLastError("fused_add: num_args should be 2");
return -1;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!((1 == ((int32_t)arg0_strides[0])))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
return -2;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* tensor = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((1 == ((int32_t)arg1_strides[0])))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
return -3;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_add: Expect arg[0] to be pointer");
return -4;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_add: Expect arg[1] to be pointer");
return -5;
  }
  if (!((1 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 1");
return -6;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
return -7;
  }
  if (!((((int32_t)arg0_shape[0]) == 10))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
return -8;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
return -9;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 1");
return -10;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
return -11;
  }
  if (!((((int32_t)arg1_shape[0]) == 10))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
return -12;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
return -13;
  }
  for (int32_t ax0 = 0; ax0 < 10; ++ax0) {
    tensor[ax0] = (placeholder[ax0] + 1.000000e+00f);
  }
  return 0;
}

