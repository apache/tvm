#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "tvm_runtime.h"

int main(int argc, char** argv) {
    TVMModuleHandle syslib = tvm_runtime_create();
    int ndim = 1;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t shape[1] = {10};

    DLTensor x;
    TVMArrayHandle input = &x;
    TVMArrayAlloc(
        shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type,
        device_id, &input);

    DLTensor y;
    TVMArrayHandle output = &y;
    TVMArrayAlloc(
        shape, ndim, dtype_code, dtype_bits, dtype_lanes, device_type,
        device_id, &output);

    for (int i = 0; i < shape[0]; ++i) {
        ((float*)(input->data))[i] = i;
    }

    TVMArgs args = TVMArgs_Create(
        (TVMValue[]){{.v_handle = input}, {.v_handle = output}},
        (uint32_t[]){kTVMNDArrayHandle, kTVMNDArrayHandle}, 2);

    TVMPackedFunc func;
    TVMPackedFunc_InitModuleFunc(
        &func, syslib, "tvmgen_default_fused_add", &args);

    TVMPackedFunc_Call(&func);

    for (int i = 0; i < shape[0]; ++i) {
        printf(
            "%.0f*2=%.0f\n", ((float*)input->data)[i],
            ((float*)output->data)[i]);
    }

    TVMArrayFree(input);
    TVMArrayFree(output);
}
