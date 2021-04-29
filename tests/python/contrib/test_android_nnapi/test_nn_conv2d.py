# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import tvm.relay
import tvm.contrib.target.android_nnapi
from . import infrastructure


def test_codegen_nchw_conv2d():
    data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
    data_v = tvm.relay.var("data", data_t)
    data_a = tvm.relay.annotation.compiler_begin(data_v, "android_nnapi")
    weight_t = tvm.relay.TensorType((1, 1, 2, 2), "float32")
    weight_v = tvm.relay.var("weight", weight_t)
    weight_a = tvm.relay.annotation.compiler_begin(weight_v, "android_nnapi")
    conv_c = tvm.relay.nn.conv2d(data=data_a, weight=weight_a)
    conv_a = tvm.relay.annotation.compiler_end(conv_c, "android_nnapi")
    func = tvm.relay.Function([data_v, weight_v], conv_a)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = infrastructure.annotate_for_android_nnapi(mod, 28)

    exe = tvm.relay.backend.vm.compile(
        mod, target="llvm -mtriple=aarch64-linux-android28", params={}
    )
    _, lib = exe.save()
    res = lib.imported_modules[1].get_source()

    ans = """
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <android/NeuralNetworks.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#define JSON2NNAPI_CHECK_EQ(a, b) { assert((a) == (b)); }
#define JSON2NNAPI_CHECK_NE(a, b) { assert((a) != (b)); }
class android_nnapi_0_0
{
public:
  android_nnapi_0_0()
  {
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_create(&this->model), ANEURALNETWORKS_NO_ERROR);
    this->createAnnModel();
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_finish(this->model), ANEURALNETWORKS_NO_ERROR);
#if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    uint32_t num_nnapi_devices;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDeviceCount(&num_nnapi_devices), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksDevice * nnapi_fallback_dev;
    for (int i = 0; i < num_nnapi_devices; i++)
    {
      JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDevice(i, &nnapi_fallback_dev), ANEURALNETWORKS_NO_ERROR);
      int32_t dev_type;
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksDevice_getType(nnapi_fallback_dev, &dev_type), ANEURALNETWORKS_NO_ERROR);
      if (dev_type == ANEURALNETWORKS_DEVICE_CPU)
      {
        break;
      }
    }
    {
      const ANeuralNetworksDevice * const dev_list[] = { nnapi_fallback_dev };
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_createForDevices(this->model, dev_list, 1, &this->compilation), ANEURALNETWORKS_NO_ERROR);
    }
#else // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_create(this->model, &this->compilation), ANEURALNETWORKS_NO_ERROR);
#endif // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_finish(this->compilation), ANEURALNETWORKS_NO_ERROR);
  }
  ~android_nnapi_0_0()
  {
    ANeuralNetworksCompilation_free(this->compilation);
    ANeuralNetworksModel_free(this->model);
    for (const auto &t: this->memories_)
    {
      ANeuralNetworksMemory_free(std::get< 1 >(t));
      close(std::get< 0 >(t));
    }
  }
  void createAnnModel()
  {
    ANeuralNetworksOperandType tensor0;
    tensor0.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor0.scale = 0.f;
    tensor0.zeroPoint = 0;
    tensor0.dimensionCount = 4;
    static uint32_t tensor0_dims[4] = {1, 1, 4, 4};
    tensor0.dimensions = tensor0_dims;
    ANeuralNetworksOperandType tensor1;
    tensor1.type = ANEURALNETWORKS_TENSOR_INT32;
    tensor1.scale = 0.f;
    tensor1.zeroPoint = 0;
    tensor1.dimensionCount = 1;
    static uint32_t tensor1_dims[1] = {4};
    tensor1.dimensions = tensor1_dims;
    ANeuralNetworksOperandType tensor2;
    tensor2.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor2.scale = 0.f;
    tensor2.zeroPoint = 0;
    tensor2.dimensionCount = 4;
    static uint32_t tensor2_dims[4] = {1, 4, 4, 1};
    tensor2.dimensions = tensor2_dims;
    ANeuralNetworksOperandType tensor3;
    tensor3.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor3.scale = 0.f;
    tensor3.zeroPoint = 0;
    tensor3.dimensionCount = 4;
    static uint32_t tensor3_dims[4] = {1, 1, 2, 2};
    tensor3.dimensions = tensor3_dims;
    ANeuralNetworksOperandType tensor4;
    tensor4.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor4.scale = 0.f;
    tensor4.zeroPoint = 0;
    tensor4.dimensionCount = 4;
    static uint32_t tensor4_dims[4] = {1, 2, 2, 1};
    tensor4.dimensions = tensor4_dims;
    ANeuralNetworksOperandType tensor5;
    tensor5.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor5.scale = 0.f;
    tensor5.zeroPoint = 0;
    tensor5.dimensionCount = 1;
    static uint32_t tensor5_dims[1] = {1};
    tensor5.dimensions = tensor5_dims;
    ANeuralNetworksOperandType scalar0;
    scalar0.type = ANEURALNETWORKS_INT32;
    scalar0.scale = 0.f;
    scalar0.zeroPoint = 0;
    scalar0.dimensionCount = 0;
    scalar0.dimensions = NULL;
    ANeuralNetworksOperandType tensor6;
    tensor6.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor6.scale = 0.f;
    tensor6.zeroPoint = 0;
    tensor6.dimensionCount = 4;
    static uint32_t tensor6_dims[4] = {1, 3, 3, 1};
    tensor6.dimensions = tensor6_dims;
    ANeuralNetworksOperandType tensor7;
    tensor7.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor7.scale = 0.f;
    tensor7.zeroPoint = 0;
    tensor7.dimensionCount = 4;
    static uint32_t tensor7_dims[4] = {1, 1, 3, 3};
    tensor7.dimensions = tensor7_dims;
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 0
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor1
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 1
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor2
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 2
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor3
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 3
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor1
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 4
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor4
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 5
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor5
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 6
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 7
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 8
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 9
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 10
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 11
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 12
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 13
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor6
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 14
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor1
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 15
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor7
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 16
    static int32_t const_val0[4] = {0, 2, 3, 1};
    static float const_val1[1] = {0.0};
    static int32_t const_val2 = 0;
    static int32_t const_val3 = 1;
    static int32_t const_val4 = ANEURALNETWORKS_FUSED_NONE;
    static int32_t const_val5[4] = {0, 3, 1, 2};
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        1,
        const_val0,
        sizeof(const_val0)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        4,
        const_val0,
        sizeof(const_val0)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        6,
        const_val1,
        sizeof(const_val1)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        7,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        8,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        9,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        10,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        11,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        12,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        13,
        &const_val4,
        sizeof(const_val4)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        15,
        const_val5,
        sizeof(const_val5)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    {
      static uint32_t inputIndexes[2] = {0, 1};
      static uint32_t outputIndexes[1] = {2};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_TRANSPOSE,
          2,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    {
      static uint32_t inputIndexes[2] = {3, 4};
      static uint32_t outputIndexes[1] = {5};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_TRANSPOSE,
          2,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    {
      static uint32_t inputIndexes[2] = {14, 15};
      static uint32_t outputIndexes[1] = {16};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_TRANSPOSE,
          2,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    {
      static uint32_t inputIndexes[10] = {2, 5, 6, 8, 10, 7, 9, 12, 11, 13};
      static uint32_t outputIndexes[1] = {14};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_CONV_2D,
          10,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    static uint32_t modelInputIndexes[2] = {0, 3};
    static uint32_t modelOutputIndexes[1] = {16};
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_identifyInputsAndOutputs(
        model,
        2,
        modelInputIndexes,
        1,
        modelOutputIndexes
      ),
      ANEURALNETWORKS_NO_ERROR
    );
  }
  void execute(float* android_nnapi_0_i0, float* android_nnapi_0_i1, float* out)
  {
    ANeuralNetworksExecution* run = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_create(this->compilation, &run), ANEURALNETWORKS_NO_ERROR);
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setInput(
        run,
        0,
        nullptr,
        android_nnapi_0_i0,
        64
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setInput(
        run,
        1,
        nullptr,
        android_nnapi_0_i1,
        16
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setOutput(
        run,
        0,
        nullptr,
        out,
        36
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    ANeuralNetworksEvent* run_end = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_startCompute(run, &run_end), ANEURALNETWORKS_NO_ERROR);
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksEvent_wait(run_end), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksEvent_free(run_end);
    ANeuralNetworksExecution_free(run);
  }
private:
  ANeuralNetworksModel* model = nullptr;
  ANeuralNetworksCompilation* compilation = nullptr;
  std::vector< std::tuple< int, ANeuralNetworksMemory* > > memories_;
};

void android_nnapi_0_(float* android_nnapi_0_i0, float* android_nnapi_0_i1, float* out0) {
  float * buf_0 = static_cast< float * >(::std::malloc(36));

  static android_nnapi_0_0 android_nnapi_0_0_instance; android_nnapi_0_0_instance.execute(reinterpret_cast< float * >(android_nnapi_0_i0), reinterpret_cast< float * >(android_nnapi_0_i1), buf_0);

  memcpy(out0, buf_0, sizeof(float) * 9);
  free(buf_0);
}

int android_nnapi_0_wrapper_(DLTensor* arg0,
        DLTensor* arg1,
        DLTensor* out0) {
  android_nnapi_0_((float*)(arg0->data),
  (float*)(arg1->data),
  (float*)(out0->data));
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t android_nnapi_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  android_nnapi_0_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif
"""
    infrastructure.verify_codegen_eq(res, ans)


def test_codegen_nchw_conv2d_on_api29():
    data_t = tvm.relay.TensorType((1, 1, 4, 4), "float32")
    data_v = tvm.relay.var("data", data_t)
    data_a = tvm.relay.annotation.compiler_begin(data_v, "android_nnapi")
    weight_t = tvm.relay.TensorType((1, 1, 2, 2), "float32")
    weight_v = tvm.relay.var("weight", weight_t)
    weight_a = tvm.relay.annotation.compiler_begin(weight_v, "android_nnapi")
    conv_c = tvm.relay.nn.conv2d(data=data_a, weight=weight_a)
    conv_a = tvm.relay.annotation.compiler_end(conv_c, "android_nnapi")
    func = tvm.relay.Function([data_v, weight_v], conv_a)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.relay.transform.PartitionGraph()(mod)
    mod = infrastructure.annotate_for_android_nnapi(mod, 29)

    exe = tvm.relay.backend.vm.compile(
        mod, target="llvm -mtriple=aarch64-linux-android29", params={}
    )
    _, lib = exe.save()
    res = lib.imported_modules[1].get_source()

    ans = """
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <android/NeuralNetworks.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#define JSON2NNAPI_CHECK_EQ(a, b) { assert((a) == (b)); }
#define JSON2NNAPI_CHECK_NE(a, b) { assert((a) != (b)); }
class android_nnapi_0_0
{
public:
  android_nnapi_0_0()
  {
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_create(&this->model), ANEURALNETWORKS_NO_ERROR);
    this->createAnnModel();
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_finish(this->model), ANEURALNETWORKS_NO_ERROR);
#if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    uint32_t num_nnapi_devices;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDeviceCount(&num_nnapi_devices), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksDevice * nnapi_fallback_dev;
    for (int i = 0; i < num_nnapi_devices; i++)
    {
      JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDevice(i, &nnapi_fallback_dev), ANEURALNETWORKS_NO_ERROR);
      int32_t dev_type;
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksDevice_getType(nnapi_fallback_dev, &dev_type), ANEURALNETWORKS_NO_ERROR);
      if (dev_type == ANEURALNETWORKS_DEVICE_CPU)
      {
        break;
      }
    }
    {
      const ANeuralNetworksDevice * const dev_list[] = { nnapi_fallback_dev };
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_createForDevices(this->model, dev_list, 1, &this->compilation), ANEURALNETWORKS_NO_ERROR);
    }
#else // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_create(this->model, &this->compilation), ANEURALNETWORKS_NO_ERROR);
#endif // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_finish(this->compilation), ANEURALNETWORKS_NO_ERROR);
  }
  ~android_nnapi_0_0()
  {
    ANeuralNetworksCompilation_free(this->compilation);
    ANeuralNetworksModel_free(this->model);
    for (const auto &t: this->memories_)
    {
      ANeuralNetworksMemory_free(std::get< 1 >(t));
      close(std::get< 0 >(t));
    }
  }
  void createAnnModel()
  {
    ANeuralNetworksOperandType tensor0;
    tensor0.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor0.scale = 0.f;
    tensor0.zeroPoint = 0;
    tensor0.dimensionCount = 4;
    static uint32_t tensor0_dims[4] = {1, 1, 4, 4};
    tensor0.dimensions = tensor0_dims;
    ANeuralNetworksOperandType tensor1;
    tensor1.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor1.scale = 0.f;
    tensor1.zeroPoint = 0;
    tensor1.dimensionCount = 4;
    static uint32_t tensor1_dims[4] = {1, 1, 2, 2};
    tensor1.dimensions = tensor1_dims;
    ANeuralNetworksOperandType tensor2;
    tensor2.type = ANEURALNETWORKS_TENSOR_INT32;
    tensor2.scale = 0.f;
    tensor2.zeroPoint = 0;
    tensor2.dimensionCount = 1;
    static uint32_t tensor2_dims[1] = {4};
    tensor2.dimensions = tensor2_dims;
    ANeuralNetworksOperandType tensor3;
    tensor3.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor3.scale = 0.f;
    tensor3.zeroPoint = 0;
    tensor3.dimensionCount = 4;
    static uint32_t tensor3_dims[4] = {1, 2, 2, 1};
    tensor3.dimensions = tensor3_dims;
    ANeuralNetworksOperandType tensor4;
    tensor4.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor4.scale = 0.f;
    tensor4.zeroPoint = 0;
    tensor4.dimensionCount = 1;
    static uint32_t tensor4_dims[1] = {1};
    tensor4.dimensions = tensor4_dims;
    ANeuralNetworksOperandType scalar0;
    scalar0.type = ANEURALNETWORKS_INT32;
    scalar0.scale = 0.f;
    scalar0.zeroPoint = 0;
    scalar0.dimensionCount = 0;
    scalar0.dimensions = NULL;
    ANeuralNetworksOperandType scalar1;
    scalar1.type = ANEURALNETWORKS_BOOL;
    scalar1.scale = 0.f;
    scalar1.zeroPoint = 0;
    scalar1.dimensionCount = 0;
    scalar1.dimensions = NULL;
    ANeuralNetworksOperandType tensor5;
    tensor5.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    tensor5.scale = 0.f;
    tensor5.zeroPoint = 0;
    tensor5.dimensionCount = 4;
    static uint32_t tensor5_dims[4] = {1, 1, 3, 3};
    tensor5.dimensions = tensor5_dims;
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 0
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor1
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 1
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor2
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 2
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor3
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 3
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor4
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 4
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 5
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 6
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 7
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 8
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 9
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 10
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 11
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar1
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 12
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 13
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &scalar0
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 14
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_addOperand(
        model,
        &tensor5
      ),
      ANEURALNETWORKS_NO_ERROR
    ); // Operand 15
    static int32_t const_val0[4] = {0, 2, 3, 1};
    static float const_val1[1] = {0.0};
    static int32_t const_val2 = 0;
    static int32_t const_val3 = 1;
    static int32_t const_val4 = ANEURALNETWORKS_FUSED_NONE;
    static bool const_val5 = true;
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        2,
        const_val0,
        sizeof(const_val0)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        4,
        const_val1,
        sizeof(const_val1)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        5,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        6,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        7,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        8,
        &const_val2,
        sizeof(const_val2)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        9,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        10,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        11,
        &const_val4,
        sizeof(const_val4)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        12,
        &const_val5,
        sizeof(const_val5)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        13,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_setOperandValue(
        model,
        14,
        &const_val3,
        sizeof(const_val3)
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    {
      static uint32_t inputIndexes[2] = {1, 2};
      static uint32_t outputIndexes[1] = {3};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_TRANSPOSE,
          2,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    {
      static uint32_t inputIndexes[13] = {0, 3, 4, 6, 8, 5, 7, 10, 9, 11, 12, 14, 13};
      static uint32_t outputIndexes[1] = {15};
      JSON2NNAPI_CHECK_EQ(
        ANeuralNetworksModel_addOperation(
          model,
          ANEURALNETWORKS_CONV_2D,
          13,
          inputIndexes,
          1,
          outputIndexes
        ),
        ANEURALNETWORKS_NO_ERROR
      );
    }
    static uint32_t modelInputIndexes[2] = {0, 1};
    static uint32_t modelOutputIndexes[1] = {15};
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksModel_identifyInputsAndOutputs(
        model,
        2,
        modelInputIndexes,
        1,
        modelOutputIndexes
      ),
      ANEURALNETWORKS_NO_ERROR
    );
  }
  void execute(float* android_nnapi_0_i0, float* android_nnapi_0_i1, float* out)
  {
    ANeuralNetworksExecution* run = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_create(this->compilation, &run), ANEURALNETWORKS_NO_ERROR);
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setInput(
        run,
        0,
        nullptr,
        android_nnapi_0_i0,
        64
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setInput(
        run,
        1,
        nullptr,
        android_nnapi_0_i1,
        16
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    JSON2NNAPI_CHECK_EQ(
      ANeuralNetworksExecution_setOutput(
        run,
        0,
        nullptr,
        out,
        36
      ),
      ANEURALNETWORKS_NO_ERROR
    );
    ANeuralNetworksEvent* run_end = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_startCompute(run, &run_end), ANEURALNETWORKS_NO_ERROR);
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksEvent_wait(run_end), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksEvent_free(run_end);
    ANeuralNetworksExecution_free(run);
  }
private:
  ANeuralNetworksModel* model = nullptr;
  ANeuralNetworksCompilation* compilation = nullptr;
  std::vector< std::tuple< int, ANeuralNetworksMemory* > > memories_;
};

void android_nnapi_0_(float* android_nnapi_0_i0, float* android_nnapi_0_i1, float* out0) {
  float * buf_0 = static_cast< float * >(::std::malloc(36));

  static android_nnapi_0_0 android_nnapi_0_0_instance; android_nnapi_0_0_instance.execute(reinterpret_cast< float * >(android_nnapi_0_i0), reinterpret_cast< float * >(android_nnapi_0_i1), buf_0);

  memcpy(out0, buf_0, sizeof(float) * 9);
  free(buf_0);
}

int android_nnapi_0_wrapper_(DLTensor* arg0,
        DLTensor* arg1,
        DLTensor* out0) {
  android_nnapi_0_((float*)(arg0->data),
  (float*)(arg1->data),
  (float*)(out0->data));
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t android_nnapi_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  android_nnapi_0_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif
"""
    infrastructure.verify_codegen_eq(res, ans)


if __name__ == "__main__":
    test_codegen_nchw_conv2d()
    test_codegen_nchw_conv2d_on_api29()
