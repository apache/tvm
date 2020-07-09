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
 * \file src/runtime/contrib/acl/acl_kernel.cc
 * \brief TVM compatible wrappers for ACL kernels.
 */

#include "acl_kernel.h"

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <string>

namespace tvm {
namespace runtime {
namespace contrib {
namespace acl {

CachedLayer::CachedLayer(const api::JSONSubGraph& function, const std::vector<NDArray>& constants,
                         ACLAllocator* allocator,
                         const std::shared_ptr<acl::MemoryManagerOnDemand>& mm)
    : constants_(constants), allocator_(allocator) {
  api::JSONOp op = function.op;
  // Make tensors
  int const_tensor_idx = 0;
  for (const auto& it : op.inputs) {
    if (it.type == "const") {
      this->function_.const_inputs.push_back(MakeTensor(it, constants[const_tensor_idx++]->data));
    } else if (it.type == "var") {
      this->function_.inputs.push_back(MakeTensor(it));
    } else {
      LOG(FATAL) << "Unsupported tensor type";
    }
  }
  for (const auto& it : op.outputs) {
    this->function_.outputs.push_back(MakeTensor(it));
  }
  // Create layer
  if (op.name == "conv2d") {
    CreateConvolution2DLayer(&this->function_, function.op, mm);
    this->is_mm_ = true;
  } else if (op.name == "max_pool") {
    CreateMaxPoolLayer(&this->function_, function.op);
  } else if (op.name == "reshape") {
    CreateReshapeLayer(&this->function_, function.op);
  } else {
    LOG(FATAL) << "Operator not yet supported";
  }
  // Prepare function
  this->function_.function->prepare();
}

bool CachedLayer::Inference(const std::vector<DLTensor*>& inputs,
                            const std::vector<DLTensor*>& outputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    CheckACLError(function_.inputs[i].allocator()->import_memory(inputs[i]->data));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    CheckACLError(function_.outputs[i].allocator()->import_memory(outputs[i]->data));
  }

  this->function_.function->run();
  return true;
}

size_t CachedLayer::GetNumInputs() const { return this->function_.inputs.size(); }

void CachedLayer::CreateConvolution2DLayer(CacheItems* cache, const api::JSONOp& params,
                                           const std::shared_ptr<acl::MemoryManagerOnDemand>& mm) {
  auto padding = dmlc::get<std::vector<int>>(params.attrs.at("padding"));
  auto strides = dmlc::get<std::vector<int>>(params.attrs.at("strides"));
  auto groups = dmlc::get<int>(params.attrs.at("groups"));

  CHECK(groups == 1) << "ACL NEON Convolution only supports group size of 1";

  acl::PadStrideInfo pad_stride_info =
      acl::PadStrideInfo(strides[0], strides[1], padding[0], padding[1], padding[2], padding[3],
                         acl::DimensionRoundingType::FLOOR);
  acl::ActivationLayerInfo act_info = acl::ActivationLayerInfo();
  if (params.attrs.find("activation_type") != params.attrs.end()) {
    auto activation_function = dmlc::get<std::string>(params.attrs.at("activation_type"));

    if (activation_function == "relu") {
      act_info = acl::ActivationLayerInfo(acl::ActivationLayerInfo::ActivationFunction::RELU);
    } else {
      LOG(FATAL) << "Unsupported activation function";
    }
  }

  auto function = std::make_shared<acl::NEConvolutionLayer>(mm);
  function->configure(&cache->inputs[0], &cache->const_inputs[0],
                      cache->const_inputs.size() > 1 ? &cache->const_inputs[1] : nullptr,
                      &cache->outputs[0], pad_stride_info, acl::WeightsInfo(), acl::Size2D(1U, 1U),
                      act_info);

  cache->function = function;
}

void CachedLayer::CreateMaxPoolLayer(CacheItems* cache, const api::JSONOp& params) {
  auto padding = dmlc::get<std::vector<int>>(params.attrs.at("padding"));
  auto strides = dmlc::get<std::vector<int>>(params.attrs.at("strides"));
  auto pool_size = dmlc::get<std::vector<int>>(params.attrs.at("pool_size"));
  auto pooling_type = dmlc::get<std::string>(params.attrs.at("pooling_type"));

  acl::PoolingType pool_type;
  if (pooling_type == "max") {
    pool_type = acl::PoolingType::MAX;
  } else {
    LOG(FATAL) << "Pooling type not supported";
  }

  acl::PadStrideInfo pad_stride_info =
      acl::PadStrideInfo(strides[0], strides[1], padding[0], padding[1], padding[2], padding[3],
                         acl::DimensionRoundingType::FLOOR);
  acl::PoolingLayerInfo pool_info = acl::PoolingLayerInfo(
      pool_type, acl::Size2D(pool_size[0], pool_size[1]), acl::DataLayout::NHWC, pad_stride_info);

  auto function = std::make_shared<acl::NEPoolingLayer>();
  function->configure(&cache->inputs[0], &cache->outputs[0], pool_info);

  cache->function = function;
}

void CachedLayer::CreateReshapeLayer(CacheItems* cache, const api::JSONOp& params) {
  auto function = std::make_shared<acl::NEReshapeLayer>();
  function->configure(&cache->inputs[0], &cache->outputs[0]);

  cache->function = function;
}

}  // namespace acl
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
