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
 * \file clml_runner.cc
 * \brief CLML model runner implementation.
 */

#include "clml_runner.h"

#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Constructor for CLMLRunner.
 * \param name is unique name for the sub graph or this CLML Runner.
 * \param args tool or utility arguments.
 * \param arg_platform_id is the OpenCL platform.
 * \param arg_context is the OpenCL context.
 * \param arg_device_id is the OpenCL device_id.
 * \param arg_queue is the OpenCL queue.
 */
CLMLRunner::CLMLRunner(std::string name, ToolArgs& args, cl_platform_id arg_platform_id,
                       cl_context arg_context, cl_device_id arg_device_id,
                       cl_command_queue arg_queue)
    : r_args(args),
      r_name(name),
      platform(arg_platform_id),
      context(arg_context),
      device_id(arg_device_id),
      queue(arg_queue) {
  LOG(INFO) << "CLMLRunner Constructor:" << name << " Input:" << r_args.input
            << " Output:" << r_args.output << " Params:" << r_args.params;
  cl_int result;

  // Query and Get CLML Interface
  static const cl_uint MAX_VERSIONS = 256;
  cl_int majorVersions[MAX_VERSIONS];
  cl_int minorVersions[MAX_VERSIONS];
  cl_uint numVersions = 0;
  result = clQueryMLInterfaceVersionsQCOM(nullptr, nullptr, 0, &numVersions);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);
  CLML_SDK_TEST_AND_EXIT(numVersions > 0u);
  CLML_SDK_TEST_AND_EXIT(numVersions <= MAX_VERSIONS);

  result = clQueryMLInterfaceVersionsQCOM(majorVersions, minorVersions, numVersions, nullptr);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  for (cl_uint i = 0; i < numVersions; ++i) {
    if (majorVersions[i] == CL_QCOM_ML_OPS_H_MAJOR_VERSION) {
      this->h_ClmlIntf = GET_ML_INTERFACE(0);
      LOG(INFO) << "CLML Target version:" << majorVersions[i];
      break;
    }
  }
  CLML_SDK_TEST_AND_EXIT(this->h_ClmlIntf != nullptr);

  result = h_ClmlIntf->clCreateMLTuningCacheQCOM(&tuning_cache);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  if (!r_args.params.empty()) {
    LOG(INFO) << "CLMLRunner Loading Params:" << r_args.params;
    npz_params = cnpy::npz_load(r_args.params);
  } else {
    LOG(INFO) << "CLMLRunner : No parameters supplied";
  }

  if (!r_args.input.empty()) {
    LOG(INFO) << "CLMLRunner Loading Inputs:" << r_args.input;
    npz_input = cnpy::npz_load(r_args.input);
  } else {
    LOG(INFO) << "CLMLRunner : No Input's given. Asuming a dry-run.";
  }
}

/*!
 * \brief Call one cycle of execution for the model.
 * \return 0 on success else error code.
 */
int CLMLRunner::Run(void) {
  LOG(INFO) << "CLMLRunner::Run :" << GetModName();
  cl_int result;

  for (size_t i = 0; i < this->function.size(); ++i) {
    result = h_ClmlIntf->clEnqueueMLOpQCOM(queue, this->function[i], this->descriptorSet, 0,
                                           nullptr, nullptr);
    CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);
  }
  if (!r_args.output.empty()) {
    for (auto it = this->outputs.begin(); it != this->outputs.end(); it++) {
      auto out_name = it->first;
      auto out_desc = it->second;
      auto dtype = outputs_dtypes[out_name];
      auto shape = outputs_shapes[out_name];
      size_t size = 1;
      for (auto si : shape) size *= si;
      if (dtype == "float32") {
        void* data = (void*)malloc(size * 4);
        CopyDataFromCLMLTensor(out_desc, data);
        LOG(INFO) << "Saving Output:" << out_name;
        cnpy::npz_save<float>(r_args.output, out_name, (float*)data, shape, "a");
        free(data);
      } else if (dtype == "int8") {
        void* data = (void*)malloc(size);
        CopyDataFromCLMLTensor(out_desc, data);
        LOG(INFO) << "Saving Output:" << out_name;
        cnpy::npz_save<int8_t>(r_args.output, out_name, (int8_t*)data, shape, "a");
        free(data);
      } else {
        LOG(WARNING) << "Unsupported dtype to dump :" << dtype;
      }
    }
  }
  return 0;
}

/*!
 * \brief Set meta information.
 * \param minfo is the meta information of the sub graph.
 */
void CLMLRunner::SetMetaInfo(std::string minfo) { this->meta_info = minfo; }

/*!
 * \brief Print the meta information.
 */
void CLMLRunner::PrintMetaInfo(void) { LOG(INFO) << "\n" << this->meta_info; }

/*!
 * \brief Copy the bytedata into tensor.
 * \param tensor is tensor descriptor to copy data.
 * \param data is pointer to bytedata.
 * \param layout is source data layout
 */
void CLMLRunner::CopyDataToCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor,
                                      void* data, cl_ml_tensor_layout_qcom layout) {
  cl_int result = 0;
  cl_event evt = nullptr;
  result = h_ClmlIntf->clEnqueueWriteMLTensorDataQCOM(this->queue, data, layout, tensor->tensor,
                                                      tensor->memory,
                                                      0,        // n waitlist
                                                      nullptr,  // waitlist
                                                      &evt);    // event
  CLML_SDK_TEST_AND_EXIT((evt != nullptr) && result == CL_SUCCESS);
}

/*!
 * \brief Copy the bytedata into tensor.
 * \param tensor is tensor descriptor to copy data.
 * \param data is pointer to bytedata.
 * \param layout is source data layout
 */
void CLMLRunner::CopyDataFromCLMLTensor(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> tensor,
                                        void* data, cl_ml_tensor_layout_qcom layout) {
  cl_int result = 0;
  cl_event readEvent = nullptr;
  // Read the output tensor
  result = h_ClmlIntf->clEnqueueReadMLTensorDataQCOM(this->queue, tensor->tensor, tensor->memory,
                                                     data, layout,
                                                     0,            // n waitlist
                                                     nullptr,      // waitlist
                                                     &readEvent);  // event
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);
  result = clWaitForEvents(1, &readEvent);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);
}

/*!
 * \brief Allocate backing memory for tensor descriptor.
 * \param pTensorMemDesc is tensor descriptor.
 * \return memory alocation status (CL_SUCCESS or error code).
 */
cl_int CLMLRunner::AllocateTensorMemory(
    std::shared_ptr<cl_ml_tensor_memory_desc_qcom> pTensorMemDesc) {
  uint32_t size = 0;
  cl_int result = CL_OUT_OF_HOST_MEMORY;
  cl_mem buffer = nullptr;

  result = h_ClmlIntf->clGetMLTensorMemorySizeQCOM(context, pTensorMemDesc->tensor, &size);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &result);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  pTensorMemDesc->memory = buffer;

  return result;
}

/*!
 * \brief Allocate memory for all tensor dectiptor in storage map.
 * Also set data for tensors given params and input numpy dumps
 */
void CLMLRunner::AllocateMemAndPopulateParams(void) {
  cl_int result;
  for (auto it = this->storage_map.begin(); it != this->storage_map.end(); it++) {
    auto node_id = it->first;
    auto tensor_desc = it->second;

    AllocateTensorMemory(tensor_desc);

    if (npz_params.find(node_id) != npz_params.end()) {
      CopyDataToCLMLTensor(tensor_desc, npz_params[node_id].data<char>());
    }

    if (npz_input.find(node_id) != npz_input.end()) {
      LOG(INFO) << "Set Input For:" << node_id;
      CopyDataToCLMLTensor(tensor_desc, npz_input[node_id].data<char>());
    }

    this->tensorMemDescs.push_back(*tensor_desc);
  }
  if (!r_args.dump_meta) {
    // Cross check all params
    for (auto nid : consts) {
      if (npz_params.find(nid) == npz_params.end()) {
        LOG(WARNING) << "Param not found in npz:" << nid;
      }
    }
  }
  // Initialize Tensor Descriptors
  result = h_ClmlIntf->clCreateMLTensorMemoryDescriptorSetQCOM(&this->descriptorSet);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  result = h_ClmlIntf->clUpdateMLTensorMemoryDescriptorSetQCOM(
      this->descriptorSet, static_cast<uint32_t>(this->tensorMemDescs.size()),
      this->tensorMemDescs.data());
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);
}

/*!
 * \brief Initializes an unused tensor.
 * It is used across operators.
 */
void CLMLRunner::MakeUnusedTensor(void) {
  cl_int result;
  cl_ml_tensor_desc_qcom desc = {};
  desc.num_dimensions = CL_TENSOR_UNUSED_QCOM;
  this->unusedTensor = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
  result = this->h_ClmlIntf->clCreateMLTensorQCOM(this->context, nullptr, &desc,
                                                  &(this->unusedTensor->tensor));
  CLML_SDK_TEST_AND_EXIT(this->unusedTensor && result == CL_SUCCESS);
}

/*!
 * \brief Convert string datatype to cl channel type.
 * \param dtype the datatype as string.
 * \return cl channel type corresponding to the datatype.
 */
cl_channel_type MakeCLDataType(const std::string& dtype) {
  if (dtype == "float32") {
    return CL_FLOAT;
  } else if (dtype == "float16") {
    return CL_HALF_FLOAT;
  } else {
    LOG(FATAL) << "Datatype: " << dtype << " unsupported by CLML runtime";
  }
  return CL_FLOAT;
}

/*!
 * \brief Map operator arthemetic mode based on data type and accumulation type.
 * \param data_type is cl channel type for computation.
 * \param acc_tpe is cl channel type for accumulation.
 * \return the arthemetic mode.
 */
cl_arithmetic_mode_qcom MakeCLArithMode(const cl_channel_type& data_type,
                                        const cl_channel_type& acc_type = CL_FLOAT) {
  if (data_type == CL_FLOAT && acc_type == CL_FLOAT) {
    return CL_ARITHMETIC_MODE_FP32_QCOM;
  } else if (data_type == CL_HALF_FLOAT && acc_type == CL_FLOAT) {
    return CL_ARITHMETIC_MODE_FP16_ACC32_QCOM;
  } else if (data_type == CL_HALF_FLOAT && acc_type == CL_HALF_FLOAT) {
    return CL_ARITHMETIC_MODE_FP16_QCOM;
  } else {
    LOG(FATAL) << "Datatype " << data_type << " unsupported by CLML runtime";
  }
}

/*!
 * \brief Creates a tensor descriptor.
 * \param shape is shape of tensor.
 * \param dtype tensor data type as string.
 * \param layout is the data layout to be used.
 * \return newly created tensor descriptor.
 */
std::shared_ptr<cl_ml_tensor_memory_desc_qcom> CLMLRunner::MakeCLMLTensor(
    std::vector<size_t> shape, std::string dtype, cl_ml_tensor_layout_qcom layout) {
  cl_int result;
  tensor_dims_t dims;
  // Make sure the tensors with dimensions less than 4 are padded with 1.
  shape.push_back(1);
  shape.push_back(1);
  shape.push_back(1);

  dims.n = shape[0];
  dims.c = shape[1];
  dims.h = shape[2];
  dims.w = shape[3];
  cl_channel_type cl_dtype = MakeCLDataType(dtype);
  auto tensor_dsc = std::make_shared<cl_ml_tensor_memory_desc_qcom>();
  cl_ml_tensor_desc_qcom desc = {
      cl_dtype, layout, dims.n, dims.c, dims.h, dims.w, 0, CL_TENSOR_DIMENSIONS_4D_QCOM, {0}};
  result =
      this->h_ClmlIntf->clCreateMLTensorQCOM(this->context, nullptr, &desc, &tensor_dsc->tensor);
  CLML_SDK_TEST_AND_EXIT(tensor_dsc->tensor && result == CL_SUCCESS);
  return tensor_dsc;
}

/*!
 * \brief Convolution2D implementation.
 * \param input_desc is input tensor descriptor.
 * \param weight_desc is the kernel as tensor descriptor.
 * \param bias_desc is bias as tensor descriptor.
 * \param output_desc is the placeholder for convolution output.
 * \param padding padding to be applied on input tensor.
 * \param dilation is convolution dilation parameter.
 * \param strides is convolution strides parameter.
 * \param groups number of groups.
 * \param mode is it normal convolution of depthwise convolution.
 * \param activation activation to be applied on result.
 * \param has_bias is bias tensor valid.
 * \param has_activation is activation to be applied.
 * \param dtype operator data type.
 */
void CLMLRunner::MakeConv2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                            std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                            std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bias_desc,
                            std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                            std::vector<cl_uint> padding, std::vector<cl_uint> dilation,
                            std::vector<cl_uint> strides, int groups, cl_convolution_mode_qcom mode,
                            cl_activation_function_qcom activation, bool has_bias, bool has_act,
                            std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_int result;
  if (CL_CONVOLUTION_MODE_CONVOLUTION_QCOM == mode) {
    CLML_SDK_TEST_AND_EXIT(groups == 1);  // CLML convolution only supports group size of 1
  } else {
    groups = 1;  // Don't need to pass groups to depthwise
  }
  cl_ml_op_activation_desc_qcom act_desc = {activation, CL_PROPAGATE_NAN_QCOM, cl_arithmetic_mode};
  cl_uint clml_padding_b[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {padding[0], padding[1]};
  cl_uint clml_padding_a[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {padding[2], padding[3]};
  cl_uint clml_strides[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {strides[0], strides[1]};
  cl_uint clml_dilation[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {dilation[0], dilation[1]};

  cl_ml_op_convolution_desc_qcom conv_desc{mode,
                                           static_cast<cl_uint>(groups),
                                           4,
                                           {clml_padding_b[0], clml_padding_b[1]},
                                           {clml_padding_a[0], clml_padding_a[1]},
                                           {clml_strides[0], clml_strides[1]},
                                           {clml_dilation[0], clml_dilation[1]},
                                           0,
                                           cl_arithmetic_mode};
  cl_ml_op_qcom op = nullptr;
  if (!has_act) {
    result = h_ClmlIntf->clCreateMLOpConvolutionForwardQCOM(
        this->context, 0, &conv_desc, input_desc->tensor, weight_desc->tensor, bias_desc->tensor,
        output_desc->tensor, &op, tuning_cache);
    CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  } else {
    result = h_ClmlIntf->clCreateMLOpFusedConvolutionActivationForwardQCOM(
        this->context, 0, &conv_desc, &act_desc, input_desc->tensor, weight_desc->tensor,
        bias_desc->tensor, nullptr, output_desc->tensor, &op, tuning_cache);
    CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  }
  this->function.push_back(op);
}

/*!
 * \brief Fused Convolution2D+BatchNorm implementation.
 * \param input_desc is input tensor descriptor.
 * \param weight_desc is the kernel as tensor descriptor.
 * \param bias_desc is bias as tensor descriptor.
 * \param output_desc is the placeholder for convolution output.
 * \param bn_scale fused batchnorm scale tensor descriptor.
 * \param bn_bias fused batchnorm scale tensor descriptor.
 * \param bn_mean fused batchnorm mean tensor descriptor.
 * \param bn_var fused batchnorm variance tensor descriptor.
 * \param bn_attrs batchnorm other attributes.
 * \param padding padding to be applied on input tensor.
 * \param dilation is convolution dilation parameter.
 * \param strides is convolution strides parameter.
 * \param groups number of groups.
 * \param mode is it normal convolution of depthwise convolution.
 * \param activation activation to be applied on result.
 * \param has_bias is bias tensor valid.
 * \param has_activation is activation to be applied.
 * \param dtype operator data type.
 */
void CLMLRunner::MakeConv2DWithBN(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bias_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_scale,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_bias,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_mean,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_var,
                                  std::vector<float> bn_attrs, std::vector<cl_uint> padding,
                                  std::vector<cl_uint> dilation, std::vector<cl_uint> strides,
                                  int groups, cl_convolution_mode_qcom mode,
                                  cl_activation_function_qcom activation, bool has_bias,
                                  bool has_act, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_int result;
  if (CL_CONVOLUTION_MODE_CONVOLUTION_QCOM == mode) {
    CLML_SDK_TEST_AND_EXIT(groups == 1);  // CLML convolution only supports group size of 1
  } else {
    groups = 1;  // Don't need to pass groups to depthwise
  }
  cl_ml_op_activation_desc_qcom act_desc = {activation, CL_PROPAGATE_NAN_QCOM, cl_arithmetic_mode};
  cl_uint clml_padding_b[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {padding[0], padding[1]};
  cl_uint clml_padding_a[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {padding[2], padding[3]};
  cl_uint clml_strides[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {strides[0], strides[1]};
  cl_uint clml_dilation[CL_ML_TENSOR_MAX_SPATIAL_DIMS_QCOM] = {dilation[0], dilation[1]};

  cl_ml_op_convolution_desc_qcom conv_desc{mode,
                                           static_cast<cl_uint>(groups),
                                           4,
                                           {clml_padding_b[0], clml_padding_b[1]},
                                           {clml_padding_a[0], clml_padding_a[1]},
                                           {clml_strides[0], clml_strides[1]},
                                           {clml_dilation[0], clml_dilation[1]},
                                           0,
                                           cl_arithmetic_mode};
  cl_ml_op_qcom op = nullptr;
  cl_ml_op_batchnorm_desc_qcom bn_desc = {CL_BATCHNORM_MODE_SPATIAL_QCOM, cl_arithmetic_mode};
  if (!has_act) {
    result = h_ClmlIntf->clCreateMLOpFusedConvolutionBatchNormForwardQCOM(
        this->context, 0, &conv_desc, &bn_desc, input_desc->tensor, weight_desc->tensor,
        bias_desc->tensor, output_desc->tensor, bn_mean->tensor, bn_var->tensor, bn_scale->tensor,
        bn_bias->tensor, &op, tuning_cache);
    CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  } else {
    result = h_ClmlIntf->clCreateMLOpFusedConvolutionBatchNormActivationForwardQCOM(
        this->context, 0, &conv_desc, &bn_desc, &act_desc, input_desc->tensor, weight_desc->tensor,
        bias_desc->tensor, output_desc->tensor, nullptr, bn_mean->tensor, bn_var->tensor,
        bn_scale->tensor, bn_bias->tensor, &op, tuning_cache);
    CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  }
  this->function.push_back(op);
}

/*!
 * \brief All types of ReLU(6) implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param relu_type the pf ReLU activation.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeRelu(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                          std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                          cl_activation_function_qcom relu_type, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;
  cl_ml_op_activation_desc_qcom act_desc = {relu_type, CL_PROPAGATE_NAN_QCOM, cl_arithmetic_mode};

  result = h_ClmlIntf->clCreateMLOpActivationForwardQCOM(
      this->context, 0, &act_desc, input_desc->tensor, this->unusedTensor->tensor,
      output_desc->tensor, &op, tuning_cache);
  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Batch Normalization operator implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param bn_scale fused batchnorm scale tensor descriptor.
 * \param bn_bias fused batchnorm scale tensor descriptor.
 * \param bn_mean fused batchnorm mean tensor descriptor.
 * \param bn_var fused batchnorm variance tensor descriptor.
 * \param bn_attrs batchnorm other attributes.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeBatchNorm(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_scale,
                               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_bias,
                               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_mean,
                               std::shared_ptr<cl_ml_tensor_memory_desc_qcom> bn_var,
                               std::vector<float> bn_attrs, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_ml_op_batchnorm_desc_qcom bn_desc = {CL_BATCHNORM_MODE_SPATIAL_QCOM, cl_arithmetic_mode};

  result = h_ClmlIntf->clCreateMLOpBatchNormForwardQCOM(
      this->context, 0, &bn_desc, input_desc->tensor, bn_mean->tensor, bn_var->tensor,
      bn_scale->tensor, bn_bias->tensor, output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief All types of Pool2D operator implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param pool_size pooling window size.
 * \param strides stride for pooling.
 * \param padding is the input padding.
 * \param pool_type is type of poling (max, avg ...etc).
 * \param dtype operator datatype.
 */
void CLMLRunner::MakePool2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                            std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                            std::vector<cl_uint> pool_size, std::vector<cl_uint> strides,
                            std::vector<cl_uint> padding, std::string pool_type,
                            std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_ml_op_pooling_desc_qcom pool_desc = {
      pool_type == "nn.max_pool2d" ? CL_POOLING_MODE_MAX_QCOM
                                   : CL_POOLING_MODE_AVERAGE_EXCLUDE_PADDING_QCOM,
      4,  // reserved
      {padding[0], padding[1]},
      {padding[2], padding[3]},
      {strides[0], strides[1]},
      {pool_size[0], pool_size[1]},
      CL_PROPAGATE_NAN_QCOM,
      cl_arithmetic_mode,
  };

  result = h_ClmlIntf->clCreateMLOpPoolingForwardQCOM(
      this->context, 0, &pool_desc, input_desc->tensor, this->unusedTensor->tensor,
      output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief All types of Global Pooling 2D operator implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param in_shape is the input tensor shape.
 * \param pool_type is the pool type (max or avg).
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeGlobalPool2D(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                                  std::vector<cl_uint> in_shape, std::string pool_type,
                                  std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;
  cl_ml_op_pooling_desc_qcom pool_desc = {
      pool_type == "nn.global_max_pool2d" ? CL_POOLING_MODE_MAX_QCOM
                                          : CL_POOLING_MODE_AVERAGE_EXCLUDE_PADDING_QCOM,
      4,  // reserved
      {0, 0},
      {0, 0},
      {1, 1},
      {in_shape[2], in_shape[3]},
      CL_PROPAGATE_NAN_QCOM,
      cl_arithmetic_mode,
  };

  result = h_ClmlIntf->clCreateMLOpPoolingForwardQCOM(
      this->context, 0, &pool_desc, input_desc->tensor, this->unusedTensor->tensor,
      output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Reshape Operator.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeReshape(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                             std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                             std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  result = h_ClmlIntf->clCreateMLOpReshapeQCOM(this->context, 0, input_desc->tensor,
                                               output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Concatenate operator implementation.
 * \param in_list list of input tensor descriptors to concatenate.
 * \param output_desc output tensor descriptor.
 * \param axis is the dimention on which we join the tensors.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeConcatenate(
    std::vector<std::shared_ptr<cl_ml_tensor_memory_desc_qcom>> in_list,
    std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, int axis, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_ml_tensor_qcom* concatInputs = new cl_ml_tensor_qcom[in_list.size()];
  for (int i = 0; i < in_list.size(); i++) {
    concatInputs[i] = in_list[i]->tensor;
  }
  cl_ml_op_concat_desc_qcom concatDesc = {1, (cl_uint)in_list.size(), cl_arithmetic_mode};
  result = h_ClmlIntf->clCreateMLOpConcatQCOM(this->context, 0, &concatDesc, concatInputs,
                                              output_desc->tensor, &op, tuning_cache);
  delete[] concatInputs;

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Dense operator implementation.
 * \param input_desc input tensor descriptor.
 * \param weight_desc weight tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param bias_desc bias tensor descriptor.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeDense(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                           std::shared_ptr<cl_ml_tensor_memory_desc_qcom> weight_desc,
                           std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                           std::vector<cl_uint> in_shape, std::vector<cl_uint> wt_shape,
                           std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;
  cl_gemm_transform_qcom b_transform = CL_GEMM_TRANSFORM_NONE_QCOM;

  if (in_shape[1] == wt_shape[1]) {
    b_transform = CL_GEMM_TRANSFORM_TRANSPOSE_QCOM;
  }

  cl_ml_op_gemm_desc_qcom gemmDesc = {in_shape[0],                  // m
                                      wt_shape[0],                  // n
                                      wt_shape[1],                  // k
                                      CL_GEMM_TRANSFORM_NONE_QCOM,  // A transform
                                      b_transform,                  // B transform
                                      {{1.0}, CL_FLOAT},            // alpha
                                      {{0.0}, CL_FLOAT},            // beta
                                      cl_arithmetic_mode};

  result =
      h_ClmlIntf->clCreateMLOpGemmQCOM(this->context, 0, &gemmDesc, input_desc->tensor,
                                       weight_desc->tensor, output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief SoftMax operator implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeSoftMax(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                             std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                             std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_ml_op_softmax_desc_qcom softmax_desc = {CL_SOFTMAX_ALGORITHM_ACCURATE_QCOM,
                                             CL_SOFTMAX_MODE_INSTANCE_QCOM, cl_arithmetic_mode};

  result = h_ClmlIntf->clCreateMLOpSoftmaxQCOM(this->context, 0, &softmax_desc, input_desc->tensor,
                                               output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief .
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param pad_mode type of padding to be applied (constant, edge, reflect ...etc).
 * \param padding amount of padding.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakePad(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                         std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                         std::string pad_mode, std::vector<cl_uint> padding, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_pad_mode_qcom clml_pad_mode = CL_PAD_MODE_CONSTANT_QCOM;
  if (pad_mode == "constant")
    clml_pad_mode = CL_PAD_MODE_CONSTANT_QCOM;
  else if (pad_mode == "edge")
    clml_pad_mode = CL_PAD_MODE_SYMMETRIC_QCOM;
  else if (pad_mode == "reflect")
    clml_pad_mode = CL_PAD_MODE_REFLECT_QCOM;
  else
    LOG(FATAL) << "Padding mode not supported by CLML:" << pad_mode;

  cl_ml_op_pad_desc_qcom pad_desc{clml_pad_mode,
                                  {0, 0},
                                  {padding[0], padding[1], padding[2], padding[3], 0, 0, 0, 0},
                                  cl_arithmetic_mode};

  result = h_ClmlIntf->clCreateMLOpPadQCOM(this->context, 0, &pad_desc, input_desc->tensor,
                                           output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Batch Flatten operator implementation.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeBatchFlatten(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                                  std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                                  std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  result = h_ClmlIntf->clCreateMLOpReshapeQCOM(this->context, 0, input_desc->tensor,
                                               output_desc->tensor, &op, tuning_cache);
  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief Clip operator.
 * \param input_desc input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param a_max is the upper bound to clip.
 * \param a_min is the lower bound to clip.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeClip(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_desc,
                          std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc, float a_max,
                          float a_min, std::string dtype) {
  LOG(INFO) << "MakeClip called";
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_ml_op_clip_desc_qcom clip_desc = {
      CL_CLIP_BY_VALUE_QCOM, {{a_max}, CL_FLOAT}, {{a_min}, CL_FLOAT}, cl_arithmetic_mode};

  result = h_ClmlIntf->clCreateMLOpClipQCOM(this->context, 0, &clip_desc, input_desc->tensor,
                                            output_desc->tensor, &op, tuning_cache);
  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

/*!
 * \brief All types of Binary operators.
 * \param input_a first input tensor descriptor.
 * \param input_b second input tensor descriptor.
 * \param output_desc output tensor descriptor.
 * \param op_name is the binary operator.
 * \param dtype operator datatype.
 */
void CLMLRunner::MakeBinaryOp(std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_a,
                              std::shared_ptr<cl_ml_tensor_memory_desc_qcom> input_b,
                              std::shared_ptr<cl_ml_tensor_memory_desc_qcom> output_desc,
                              std::string op_name, std::string dtype) {
  cl_arithmetic_mode_qcom cl_arithmetic_mode = MakeCLArithMode(MakeCLDataType(dtype));
  cl_ml_op_qcom op = nullptr;
  cl_int result;

  cl_binary_op_qcom binary_op = CL_TENSOR_OP_ADD_QCOM;
  if (op_name == "subtract")
    binary_op = CL_TENSOR_OP_SUB_QCOM;
  else if (op_name == "multiply")
    binary_op = CL_TENSOR_OP_MUL_QCOM;
  else if (op_name == "divide")
    binary_op = CL_TENSOR_OP_DIV_QCOM;
  else if (op_name == "minimum")
    binary_op = CL_TENSOR_OP_MIN_QCOM;
  else if (op_name == "maximum")
    binary_op = CL_TENSOR_OP_MAX_QCOM;
  cl_ml_op_binary_desc_qcom add_desc = {
      binary_op, {{1.0}, CL_FLOAT}, {{1.0}, CL_FLOAT}, {{0.0}, CL_FLOAT}, cl_arithmetic_mode};

  result =
      h_ClmlIntf->clCreateMLOpBinaryQCOM(this->context, 0, &add_desc, input_a->tensor,
                                         input_b->tensor, output_desc->tensor, &op, tuning_cache);

  CLML_SDK_TEST_AND_EXIT(op && result == CL_SUCCESS);
  this->function.push_back(op);
}

}  // namespace runtime
}  // namespace tvm
