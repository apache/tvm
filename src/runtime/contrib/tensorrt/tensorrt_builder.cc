/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_builder.cc
 * \brief The TensorRTBuilder class can be used to convert a JSONRuntime graph into a TRT engine
 * which can be used for inference.
 */

#include "tensorrt_builder.h"

#include <tvm/runtime/ndarray.h>

#include <memory>
#include <string>

#include "tensorrt_logger.h"
#include "tensorrt_ops.h"
#include "tensorrt_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

TensorRTBuilder::TensorRTBuilder(TensorRTLogger* logger,
                                 const std::vector<const DLTensor*>& data_entry,
                                 size_t max_workspace_size, bool use_implicit_batch, bool use_fp16,
                                 int batch_size, nvinfer1::IInt8Calibrator* calibrator)
    : data_entry_(data_entry),
      max_workspace_size_(max_workspace_size),
      use_implicit_batch_(use_implicit_batch),
      use_fp16_(use_fp16),
      use_int8_(false),
      batch_size_(batch_size),
      calibrator_(calibrator) {
  // Create TRT builder and network.
  builder_ = nvinfer1::createInferBuilder(*logger);

#if TRT_VERSION_GE(6, 0, 1)
  // Use INetworkV2.
  auto flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  if (use_implicit_batch_) {
    flags = 0U;
    builder_->setMaxBatchSize(batch_size_);
  }
  if (calibrator_ != nullptr) {
    use_int8_ = true;
  }
  network_ = builder_->createNetworkV2(flags);
#else
  builder_->setMaxBatchSize(batch_size_);
  builder_->setMaxWorkspaceSize(max_workspace_size_);
  builder_->setFp16Mode(use_fp16_);
  network_ = builder_->createNetwork();
#endif
}

nvinfer1::DataType DLDataType2NVDataType(DLDataType data_type) {
  ICHECK(data_type.code == kDLFloat && (data_type.bits == 16 || data_type.bits == 32))
      << "Invalid input Tensor type. Only float16 and float32 are supported";
  return (data_type.bits == 16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
}

void TensorRTBuilder::AddInput(int nid, uint32_t entry_id, const JSONGraphNode& node) {
  auto node_name = node.GetOpName();
  auto shapes = node.GetOpShape();
  auto dtypes = node.GetOpDataType();
  ICHECK_EQ(shapes.size(), dtypes.size());
  node_output_map_[nid] = {};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const std::string name = node_name + "_" + std::to_string(i);
    auto shape = shapes[i];
    // Remove batch dim when not in explicit batch mode.
    if (use_implicit_batch_ && shape.size() > 1) {
      shape.erase(shape.begin());
    }
    nvinfer1::Dims dims = VectorToTrtDims(shape);
    auto input_tensor = network_->addInput(name.c_str(), DLDataType2NVDataType(dtypes[i]), dims);
    node_output_map_[nid].push_back(TensorRTOpInput(input_tensor));
    network_input_names_.push_back(name);
    entry_id_map_[name] = entry_id + i;
  }
}

void TensorRTBuilder::AddConstant(int nid, const DLTensor* data) {
  nvinfer1::Weights weight = GetDLTensorAsWeights(data, kDLCPU);
  std::vector<int> shape(data->shape, data->shape + data->ndim);
  node_output_map_[nid] = {TensorRTOpInput(weight, shape)};
}

void TensorRTBuilder::AddOutput(const JSONGraphNodeEntry& node, uint32_t entry_id) {
  auto it = node_output_map_.find(node.id_);
  ICHECK(it != node_output_map_.end()) << "Output was not found.";
  auto out_tensor = it->second[node.index_].tensor;
  std::string name = "tensorrt_output_" + std::to_string(network_output_names_.size());
  // If the network is already marked as an input or output, make a copy to avoid TRT crash.
  if (out_tensor->isNetworkOutput()) {
    LOG(WARNING) << name << " is a duplicate output.";
    out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  } else if (out_tensor->isNetworkInput()) {
    LOG(WARNING) << name << " is both an input and an output.";
    out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  }
  out_tensor->setName(name.c_str());
  network_->markOutput(*out_tensor);
  network_output_names_.push_back(name);
  entry_id_map_[name] = entry_id;
}

void TensorRTBuilder::AddLayer(int nid, const JSONGraphNode& node) {
  TensorRTOpConverterParams params(network_, nid, node, &trt_weights_);
  // Look up converter.
  const std::unordered_map<std::string, std::unique_ptr<TensorRTOpConverter>>& map =
      GetOpConverters();
  auto it = map.find(params.op_name);
  ICHECK(it != map.end()) << params.op_name << ": Unsupported operator";
  const TensorRTOpConverter& converter = *it->second;
  if (!converter.variable_input_count) {
    ICHECK_EQ(node.GetInputs().size(), converter.input_types.size())
        << params.op_name << ": Mismatched input sizes";
  }
  // Get inputs.
  for (size_t i = 0; i < node.GetInputs().size(); ++i) {
    auto in_node = node.GetInputs()[i];
    auto it = node_output_map_.find(in_node.id_);
    ICHECK(it != node_output_map_.end()) << params.op_name << ": Input was not found";
    auto input = it->second[in_node.index_];
    if (!converter.variable_input_count) {
      if (converter.input_types[i] == kTensor && input.type == kWeight) {
        input = TensorRTOpInput(GetInputAsTensor(input));
      } else if (converter.input_types[i] == kWeight && input.type == kTensor) {
        LOG(FATAL) << params.op_name << ": Input " << i << " must be a constant.";
      }
    }
    params.inputs.push_back(input);
  }

  // Convert op to TRT.
  converter.Convert(&params);

  // Get outputs.
  node_output_map_[nid] = {};
  std::vector<DLDataType> dtype = node.GetOpDataType();
  ICHECK_EQ(params.outputs.size(), dtype.size()) << params.op_name << ": Mismatched output sizes";
  for (size_t i = 0; i < params.outputs.size(); ++i) {
    auto out = params.outputs[i];
    out->setType(DLDataType2NVDataType(dtype[i]));
    node_output_map_[nid].push_back(TensorRTOpInput(out));
  }
}

TensorRTEngineAndContext TensorRTBuilder::BuildEngine() {
  // Process graph to create INetworkDefinition.
// Build engine.
#if TRT_VERSION_GE(6, 0, 1)
  config_ = builder_->createBuilderConfig();
  config_->setMaxWorkspaceSize(max_workspace_size_);
  if (use_fp16_) {
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  if (use_int8_) {
    config_->setFlag(nvinfer1::BuilderFlag::kINT8);
    ICHECK(calibrator_);
    config_->setInt8Calibrator(calibrator_);
    LOG(INFO) << "config finishes setting up calibrator as INT8 mode ... ";
  }

  // Add profiles.
  if (!use_implicit_batch_) {
    auto profile = builder_->createOptimizationProfile();
    for (int i = 0; i < network_->getNbInputs(); ++i) {
      auto name = network_->getInput(i)->getName();
      const uint32_t entry_id = entry_id_map_[name];
      std::vector<int64_t> shape(data_entry_[entry_id]->shape,
                                 data_entry_[entry_id]->shape + data_entry_[entry_id]->ndim);
      auto dims = VectorToTrtDims(shape);

      profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims);
      // Set minimum batch size to 1 when dynamic batching is used.
      if (network_->getInput(i)->getDimensions().nbDims >= 1 &&
          network_->getInput(i)->getDimensions().d[0] == -1) {
        dims.d[0] = 1;
      }
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims);
    }
    config_->addOptimizationProfile(profile);
  }
  nvinfer1::ICudaEngine* engine = builder_->buildEngineWithConfig(*network_, *config_);
#else
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
#endif
  ICHECK_EQ(engine->getNbBindings(), network_input_names_.size() + network_output_names_.size());
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  CleanUp();

  ICHECK(engine);
  ICHECK(context);

  return {engine, context, network_input_names_, network_output_names_};
}

nvinfer1::Weights TensorRTBuilder::GetDLTensorAsWeights(const DLTensor* dptr,
                                                        DLDeviceType src_device) {
  ICHECK_EQ(dptr->device.device_type, src_device);
  ICHECK((dptr->dtype.bits != 16 || dptr->dtype.bits != 32))
      << "Invalid input Tensor type. Float16 and Float32 are supported";
  const auto trt_dtype = (static_cast<int>(dptr->dtype.bits) == 16) ? nvinfer1::DataType::kHALF
                                                                    : nvinfer1::DataType::kFLOAT;

  const size_t weight_bytes = GetDataSize(*dptr);
  nvinfer1::Weights weight{trt_dtype, nullptr, 0};
  size_t count = 1;
  for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
    count *= dptr->shape[i];
  }
  weight.count = count;
  weight.values = new float[count];
  ICHECK_EQ(TVMArrayCopyToBytes(const_cast<DLTensor*>(dptr), const_cast<void*>(weight.values),
                                weight_bytes),
            0)
      << TVMGetLastError();
  trt_weights_.push_back(weight);
  return weight;
}

nvinfer1::ITensor* TensorRTBuilder::GetInputAsTensor(const TensorRTOpInput& input) {
  if (input.type == kTensor) return input.tensor;
  auto shape = input.weight_shape;
  // Remove batch dim when not in explicit batch mode.
  // Example:
  // x = Relay dims (1, 32, 224, 224) which becomes TRT Dims (32, 224, 224)
  // y = Relay dims (1, 32)
  // z = add(x, y)
  // y needs to have TRT dims (32,), otherwise broadcasting will result in z having
  // TRT Dims(1, 32, 224, 224) when it should be (32, 224, 224).
  if (use_implicit_batch_ && shape.size() > 1 && shape[0] == 1) {
    shape.erase(shape.begin());
  }
  return network_->addConstant(VectorToTrtDims(shape), input.weight)->getOutput(0);
}

void TensorRTBuilder::CleanUp() {
  VLOG(1) << "Destroying TensorRT network";
  ICHECK(network_);
  network_->destroy();
  network_ = nullptr;

#if TRT_VERSION_GE(6, 0, 1)
  VLOG(1) << "Destroying TensorRT config";
  ICHECK(config_);
  config_->destroy();
  config_ = nullptr;
#endif

  VLOG(1) << "Destroying TensorRT builder";
  ICHECK(builder_);
  builder_->destroy();
  builder_ = nullptr;

  VLOG(1) << "Destroying TensorRT weights";
  for (auto weight : trt_weights_) {
    ICHECK(weight.values);
    if (weight.type == nvinfer1::DataType::kFLOAT || weight.type == nvinfer1::DataType::kHALF) {
      delete[] static_cast<const float*>(weight.values);
    } else {
      delete[] static_cast<const uint16_t*>(weight.values);
    }
  }
  trt_weights_.clear();
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
