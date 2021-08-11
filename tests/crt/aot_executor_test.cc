/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <dlpack/dlpack.h>
#include <gtest/gtest.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/internal/aot_executor/aot_executor.h>

int test_run_func(TVMValue* args, int* arg_type_ids, int num_args, TVMValue* out_ret_value,
                  int* out_ret_tcode, void* resource_handle) {
  return kTvmErrorNoError;
}

TEST(AOTRuntime, NoOp) {
  const tvm_model_t test_model = {
      .num_input_tensors = 0,
      .num_output_tensors = 0,
      .run_func = &test_run_func,
  };

  ASSERT_EQ(kTvmErrorNoError, tvm_runtime_run(&test_model, NULL, NULL));
}

int32_t error_run_func(TVMValue* args, int* arg_type_ids, int32_t num_args, TVMValue* out_ret_value,
                       int* out_ret_tcode, void* resource_handle) {
  return kTvmErrorPlatformNoMemory;
}

TEST(AOTRuntime, Error) {
  const tvm_model_t error_model = {
      .num_input_tensors = 0,
      .num_output_tensors = 0,
      .run_func = &error_run_func,
  };

  ASSERT_EQ(kTvmErrorPlatformNoMemory, tvm_runtime_run(&error_model, NULL, NULL));
}

int32_t identity_run_func(TVMValue* args, int* arg_type_ids, int32_t num_args,
                          TVMValue* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* T_id = (((DLTensor*)arg1)[0].data);
  ((uint32_t*)T_id)[(0)] = ((uint32_t*)placeholder)[(0)];
  return kTvmErrorNoError;
}

TEST(AOTRuntime, Identity) {
  const tvm_model_t identity_model = {
      .num_input_tensors = 1,
      .num_output_tensors = 1,
      .run_func = &identity_run_func,
  };

  uint32_t inputs1[1] = {404};
  void* inputs[] = {inputs1};
  uint32_t outputs1[1];
  void* outputs[] = {outputs1};

  ASSERT_EQ(kTvmErrorNoError, tvm_runtime_run(&identity_model, inputs, outputs));
  ASSERT_EQ(outputs1[0], 404U);
}

int32_t add_run_func(TVMValue* args, int* arg_type_ids, int32_t num_args, TVMValue* out_ret_value,
                     int* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* T_add = (((DLTensor*)arg1)[0].data);
  ((uint32_t*)T_add)[(0)] = ((uint32_t*)placeholder)[(0)] + ((uint32_t*)placeholder)[(1)];
  return kTvmErrorNoError;

  return kTvmErrorNoError;
}

TEST(AOTRuntime, Add) {
  const tvm_model_t add_model = {
      .num_input_tensors = 1,
      .num_output_tensors = 1,
      .run_func = &add_run_func,
  };

  uint32_t inputs1[2] = {404, 500};
  void* inputs[] = {inputs1};
  uint32_t outputs1[1];
  void* outputs[] = {outputs1};

  ASSERT_EQ(kTvmErrorNoError, tvm_runtime_run(&add_model, inputs, outputs));
  ASSERT_EQ(outputs1[0], 904U);
}

int32_t multiple_inputs_run_func(TVMValue* args, int* arg_type_ids, int32_t num_args,
                                 TVMValue* out_ret_value, int* out_ret_tcode,
                                 void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* T_add = (((DLTensor*)arg2)[0].data);
  ((uint32_t*)T_add)[(0)] = ((uint32_t*)placeholder)[(0)] + ((uint32_t*)placeholder)[(1)] +
                            ((uint32_t*)placeholder1)[(0)] + ((uint32_t*)placeholder1)[(1)];
  return kTvmErrorNoError;
}

TEST(AOTRuntime, MultipleInputs) {
  const tvm_model_t multiple_inputs_model = {
      .num_input_tensors = 2,
      .num_output_tensors = 1,
      .run_func = &multiple_inputs_run_func,
  };

  uint32_t inputs1[2] = {404, 500};
  uint32_t inputs2[2] = {200, 202};
  void* inputs[] = {inputs1, inputs2};

  uint32_t outputs1[1];
  void* outputs[] = {outputs1};

  ASSERT_EQ(kTvmErrorNoError, tvm_runtime_run(&multiple_inputs_model, inputs, outputs));
  ASSERT_EQ(outputs1[0], 1306U);
}

int32_t multiple_outputs_run_func(TVMValue* args, int* arg_type_ids, int32_t num_args,
                                  TVMValue* out_ret_value, int* out_ret_tcode,
                                  void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* T_split1 = (((DLTensor*)arg1)[0].data);
  void* T_split2 = (((DLTensor*)arg2)[0].data);
  ((uint32_t*)T_split1)[(0)] = ((uint32_t*)placeholder)[(0)];
  ((uint32_t*)T_split2)[(0)] = ((uint32_t*)placeholder)[(1)];
  return kTvmErrorNoError;
}

TEST(AOTRuntime, MultipleOutputs) {
  const tvm_model_t multiple_outputs_model = {
      .num_input_tensors = 1,
      .num_output_tensors = 2,
      .run_func = &multiple_outputs_run_func,
  };

  uint32_t inputs1[2] = {404, 500};
  void* inputs[] = {inputs1};

  uint32_t outputs1[1];
  uint32_t outputs2[1];
  void* outputs[] = {outputs1, outputs2};

  ASSERT_EQ(kTvmErrorNoError, tvm_runtime_run(&multiple_outputs_model, inputs, outputs));
  ASSERT_EQ(outputs1[0], 404U);
  ASSERT_EQ(outputs2[0], 500U);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
