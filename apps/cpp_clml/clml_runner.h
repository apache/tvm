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

/*!
 * \file clml_runner.h
 * \brief CLML model runner.
 */
#ifndef CLML_APPS_CPP_RCLML_RUNNER_H_
#define CLML_APPS_CPP_RCLML_RUNNER_H_

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

#include <CL/cl_qcom_ml_ops.h>
#include <cnpy.h>
#include <dmlc/io.h>

#include "CL/cl.h"

#define CLML_SDK_TEST_AND_EXIT(expression)                                                      \
  {                                                                                             \
    {                                                                                           \
      int _n_ = !(expression);                                                                  \
      if (_n_) {                                                                                \
        fprintf(stderr, "Error on line %d of %s\nFailing expression: %s\n", __LINE__, __FILE__, \
                #expression);                                                                   \
        exit(1);                                                                                \
      }                                                                                         \
    }                                                                                           \
  }

#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)
#define GET_ML_INTERFACE CAT(CAT(clGetMLInterfaceV, CL_QCOM_ML_OPS_H_MAJOR_VERSION), QCOM)
#define GET_ML_API_INTERFACE CAT(CAT(CLMLInterfaceV, CL_QCOM_ML_OPS_H_MAJOR_VERSION), QCOM)

namespace tvm {
namespace runtime {

/**
 * \brief Tensor dimensions, batch, channel, height, width
 *
 */
struct tensor_dims_t {
  uint32_t n, c, h, w;
};

/*!
 * \brief Tool Arguments.
 * \arg input Numpy file for the model input
 * \arg output Numpy file name to dump the model output as numpy
 * \arg parsms Numpy file holding the params for models
 */
struct ToolArgs {
  std::string input;
  std::string output;
  std::string params;
  bool dump_meta = false;
};

/*!
 * \brief encapsulates CLML Runner functionality for the sub graph
 */
class CLMLRunner {
 public:
  /*! \brief Constructor */
  CLMLRunner(std::string name, ToolArgs& args, cl_platform_id arg_platform_id,
             cl_context arg_context, cl_device_id arg_device_id, cl_command_queue arg_queue);

  /*! \brief Returns the name for this sub graph */
  std::string GetModName(void) { return r_name; }
  /*! \brief Executes one cycle all CLML ops */
  int Run(void);
  /*! \brief set meta information */
  void SetMetaInfo(std::string minfo);
  /*! \brief Print function to show all meta information */
  void PrintMetaInfo(void);
  /*! \brief initializes the unusedTensor */
  void MakeUnusedTensor(void);
  /*! \brief Copy given bytestream of data to the tensor */
  void CopyDataToCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                            cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM);
  /*! \brief Copy tensor data to data in expected layout format */
  void CopyDataFromCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor, void* data,
                              cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_NCHW_QCOM);
  /*! \brief Allocates memory for the tensor descriptor */
  cl_int AllocateTensorMemory(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> pTensorMemDesc);
  /*!
   * \brief Allocates memory for all tensor descriptor in storage map.
   * Also initializes the parameter nodes, inputs from given numpy dumps if provided.
   */
  void AllocateMemAndPopulateParams(void);
  /*! \brief Create a tensor descriptor given it's shape, dtype and layout */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> MakeCLMLTensor(
      std::vector<size_t> shape, std::string dtype = "float32",
      cl_ml_tensor_layout_qcom layout = CL_TENSOR_LAYOUT_OPTIMAL_QCOM);
  /*! \brief Conv2D layer implementattion */
  void MakeConv2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bias_desc,
                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                  std::vector<cl_uint> padding, std::vector<cl_uint> dilation,
                  std::vector<cl_uint> strides, int groups, cl_convolution_mode_qcom mode,
                  cl_activation_function_qcom activation, bool has_bias, bool has_act,
                  std::string dtype);

  /*! \brief Conv2D with Fused BatchNorm layer implementattion */
  void MakeConv2DWithBN(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bias_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_scale,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_bias,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_mean,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_var,
                        std::vector<float> bn_attrs, std::vector<cl_uint> padding,
                        std::vector<cl_uint> dilation, std::vector<cl_uint> strides, int groups,
                        cl_convolution_mode_qcom mode, cl_activation_function_qcom activation,
                        bool has_bias, bool has_act, std::string dtype);

  /*! \brief ReLU layer implementattion */
  void MakeRelu(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                cl_activation_function_qcom relu_type, std::string dtype);

  /*! \brief Batch Normalization layer implementattion */
  void MakeBatchNorm(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                     std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                     std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_scale,
                     std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_bias,
                     std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_mean,
                     std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_var,
                     std::vector<float> bn_attrs, std::string dtype);

  /*! \brief Pool2D (with all variants) layer implementattion */
  void MakePool2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                  std::vector<cl_uint> pool_size, std::vector<cl_uint> strides,
                  std::vector<cl_uint> padding, std::string pool_type, std::string dtype);

  /*! \brief GlobalPool2D (with all variants) layer implementattion */
  void MakeGlobalPool2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                        std::vector<cl_uint> in_shape, std::string pool_type, std::string dtype);

  /*! \brief Reshape layer implementattion */
  void MakeReshape(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                   std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, std::string dtype);

  /*! \brief Concatenate layer implementattion */
  void MakeConcatenate(std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> in_list,
                       std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, int axis,
                       std::string dtype);

  /*! \brief Dense layer implementattion */
  void MakeDense(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                 std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                 std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                 std::vector<cl_uint> in_shape, std::vector<cl_uint> wt_shape, std::string dtype);

  /*! \brief SoftMax layer implementattion */
  void MakeSoftMax(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                   std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, std::string dtype);

  /*! \brief Pad layer implementattion */
  void MakePad(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, std::string pad_mode,
               std::vector<cl_uint> padding, std::string dtype);

  /*! \brief Batch Flatten layer implementattion */
  void MakeBatchFlatten(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                        std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                        std::string dtype);

  /*! \brief Clip layer implementattion */
  void MakeClip(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, float a_max,
                float a_min, std::string dtype);

  /*! \brief Binary Operator (with all types) layer implementattion */
  void MakeBinaryOp(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_a,
                    std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_b,
                    std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, std::string op_name,
                    std::string dtype);

  /*! \brief Vector of created operators */
  std::vector<cl_ml_op_qcom> function;
  /*! \brief Vector of graph's input tensor descriptors */
  std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> inputs;
  /*! \brief Map of graph's output tensor descriptors with names */
  std::map<std::string, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> outputs;
  /*! \brief Map of graph's output tensor names and dtypes */
  std::map<std::string, std::string> outputs_dtypes;
  /*! \brief Map of graph's output tensor names and shapes */
  std::map<std::string, std::vector<size_t>> outputs_shapes;
  /*! \brief Overall storage map for all tensor descriptors involved */
  std::map<std::string, std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> storage_map;
  /*! \brief List of const tensor of the graph */
  std::vector<std::string> consts;
  /*! \brief List of all memory descriptor in graph */
  std::vector<cl_ml_tensor_memory_desc_qcom> tensorMemDescs;
  /*! \brief Tensor memory descriptor set */
  cl_ml_tensor_mem_desc_set_qcom descriptorSet;
  /*! \brief Unused tensor used across various ops */
  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> unusedTensor;

  /*! \brief  ML API interface */
  GET_ML_API_INTERFACE* h_ClmlIntf = nullptr;
  /*! \brief  Tuning cache object */
  cl_ml_tuningcache_qcom tuning_cache = nullptr;
  /*! \brief  Flag to inticate a tuning run */
  bool is_tuning_run;
  /*! \brief  The tuning file for loading or storing cache */
  char* tuning_file;

  /*! \brief  OpenCL platform */
  cl_platform_id platform{nullptr};
  /*! \brief  OpenCL context */
  cl_context context{nullptr};
  /*! \brief  OpenCL device */
  cl_device_id device_id{nullptr};
  /*! \brief  OpenCL Queue */
  cl_command_queue queue{nullptr};
  /*! \brief  Numpy object for params */
  cnpy::npz_t npz_params;
  /*! \brief  Numpy object for inputs */
  cnpy::npz_t npz_input;

 private:
  /*! \brief unique name for the runner */
  std::string r_name;
  /*! \brief arguments */
  ToolArgs r_args;
  /*! \brief Holds meta information from clml codegen */
  std::string meta_info;
};

}  // namespace runtime
}  // namespace tvm
#endif  // CLML_APPS_CPP_RCLML_RUNNER_H_
