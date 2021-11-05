#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tvm_runtime.h"

extern unsigned char test_xiaoai[];
extern unsigned int test_xiaoai_len;

extern unsigned char test_unknown[];
extern unsigned int test_unknown_len;

int main(int argc, char** argv) {
    TVMGraphExecutor* executor = tvm_runtime_create();
    float input_data[1 * 99 * 12] = {0};

    DLTensor input;
    input.data = input_data;
    DLDevice dev = {kDLCPU, 0};
    input.device = dev;
    input.ndim = 3;
    DLDataType dtype = {kDLFloat, 32, 1};
    input.dtype = dtype;
    input.shape = (int64_t[]){1, 99, 12};
    input.strides = NULL;
    input.byte_offset = 0;

    float output_data[1] = {0};
    DLTensor output;
    output.data = output_data;
    DLDevice out_dev = {kDLCPU, 0};
    output.device = out_dev;
    output.ndim = 2;
    DLDataType out_dtype = {kDLFloat, 32, 1};
    output.dtype = out_dtype;
    output.shape = (int64_t[]){1, 1};
    output.strides = NULL;
    output.byte_offset = 0;

    TVMGraphExecutor_SetInput(executor, "input_1", &input);

    int STEP = 99 * 12 * sizeof(float) / sizeof(char);

#define TEST(data, len)                                       \
    {                                                         \
        unsigned char* mfcc = (unsigned char*)data;           \
        int n = len / STEP;                                   \
        for (int i = 0; i < n; i++) {                         \
            memcpy(input_data, mfcc, STEP);                   \
            mfcc += STEP;                                     \
            TVMGraphExecutor_Run(executor);                   \
            TVMGraphExecutor_GetOutput(executor, 0, &output); \
            printf("output: %f\n", output_data[0]);           \
        }                                                     \
    }

    printf("\n------test xiaoai------\n");
    TEST(test_xiaoai, test_xiaoai_len);

    printf("\n------test unknown------\n");
    TEST(test_unknown, test_unknown_len);
}
