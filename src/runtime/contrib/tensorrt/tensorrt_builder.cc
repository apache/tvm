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
* \brief Contains TensorRTBuilder class which can be used to convert a relay
* program into a TRT engine which can be used for inference.
*/

#include <memory>
#include <string>

#include "tensorrt_builder.h"
#include "tensorrt_logger.h"
#include "tensorrt_ops.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace contrib {

const std::shared_ptr<
    std::unordered_map<std::string, std::shared_ptr<TrtOpConverter>>>
GetOpConverters() {
  static auto map = std::make_shared<
      std::unordered_map<std::string, std::shared_ptr<TrtOpConverter>>>();
  if (!map->empty()) return map;
  map->emplace("nn.relu", std::make_shared<ActivationOpConverter>());
  map->emplace("sigmoid", std::make_shared<ActivationOpConverter>());
  map->emplace("tanh", std::make_shared<ActivationOpConverter>());
  map->emplace("nn.batch_norm", std::make_shared<BatchNormOpConverter>());
  map->emplace("nn.softmax", std::make_shared<SoftmaxOpConverter>());
  map->emplace("nn.conv2d", std::make_shared<Conv2DOpConverter>());
  map->emplace("nn.dense", std::make_shared<DenseOpConverter>());
  map->emplace("nn.bias_add", std::make_shared<BiasAddOpConverter>());
  map->emplace("add", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("subtract", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("multiply", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("divide", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("power", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("maximum", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("minimum", std::make_shared<ElementWiseBinaryOpConverter>());
  map->emplace("nn.max_pool2d", std::make_shared<PoolingOpConverter>());
  map->emplace("nn.avg_pool2d", std::make_shared<PoolingOpConverter>());
  map->emplace("nn.global_max_pool2d",
               std::make_shared<GlobalPoolingOpConverter>());
  map->emplace("nn.global_avg_pool2d",
               std::make_shared<GlobalPoolingOpConverter>());
  map->emplace("exp", std::make_shared<UnaryOpConverter>());
  map->emplace("log", std::make_shared<UnaryOpConverter>());
  map->emplace("sqrt", std::make_shared<UnaryOpConverter>());
  map->emplace("abs", std::make_shared<UnaryOpConverter>());
  map->emplace("negative", std::make_shared<UnaryOpConverter>());
  map->emplace("nn.batch_flatten", std::make_shared<BatchFlattenOpConverter>());
  map->emplace("expand_dims", std::make_shared<ExpandDimsOpConverter>());
  map->emplace("squeeze", std::make_shared<SqueezeOpConverter>());
  map->emplace("concatenate", std::make_shared<ConcatOpConverter>());
  map->emplace("nn.conv2d_transpose",
               std::make_shared<Conv2DTransposeOpConverter>());
  map->emplace("transpose", std::make_shared<TransposeOpConverter>());
  map->emplace("reshape", std::make_shared<ReshapeOpConverter>());
  map->emplace("nn.pad", std::make_shared<PadOpConverter>());
  map->emplace("sum", std::make_shared<ReduceOpConverter>());
  map->emplace("prod", std::make_shared<ReduceOpConverter>());
  map->emplace("max", std::make_shared<ReduceOpConverter>());
  map->emplace("min", std::make_shared<ReduceOpConverter>());
  map->emplace("mean", std::make_shared<ReduceOpConverter>());
  map->emplace("nn.adaptive_max_pool2d",
               std::make_shared<AdaptivePoolingOpConverter>());
  map->emplace("nn.adaptive_avg_pool2d",
               std::make_shared<AdaptivePoolingOpConverter>());
#if TRT_VERSION_GE(5, 1, 5)
  map->emplace("clip", std::make_shared<ActivationOpConverter>());
  map->emplace("nn.leaky_relu", std::make_shared<ActivationOpConverter>());
  map->emplace("sin", std::make_shared<UnaryOpConverter>());
  map->emplace("cos", std::make_shared<UnaryOpConverter>());
  map->emplace("atan", std::make_shared<UnaryOpConverter>());
  map->emplace("ceil", std::make_shared<UnaryOpConverter>());
  map->emplace("floor", std::make_shared<UnaryOpConverter>());
  map->emplace("strided_slice", std::make_shared<StridedSliceOpConverter>());
  map->emplace("split", std::make_shared<SplitOpConverter>());
#else
  map->emplace("clip", std::make_shared<ClipLegacyOpConverter>());
#endif
#if TRT_VERSION_GE(6, 0, 1)
  map->emplace("image.resize", std::make_shared<ResizeOpConverter>());
  map->emplace("nn.upsampling", std::make_shared<UpsamplingOpConverter>());
  map->emplace("nn.conv3d", std::make_shared<Conv3DOpConverter>());
  map->emplace("nn.max_pool3d", std::make_shared<Pooling3DOpConverter>());
  map->emplace("nn.avg_pool3d", std::make_shared<Pooling3DOpConverter>());
  map->emplace("nn.conv3d_transpose", std::make_shared<Conv3DTransposeOpConverter>());
#endif
  return map;
}

TensorRTBuilder::TensorRTBuilder(runtime::TensorRTLogger* logger,
    const std::vector<DLTensor*>& args, size_t max_workspace_size, bool use_implicit_batch)
    : execution_args_(args), max_workspace_size_(max_workspace_size),
      use_implicit_batch_(use_implicit_batch) {
  // Create TRT builder and network.
  builder_ = nvinfer1::createInferBuilder(*logger);
#if TRT_VERSION_GE(6, 0, 1)
  // Use INetworkV2.
  auto flags = 1U << static_cast<uint32_t>(
      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);;
  if (use_implicit_batch_) {
    flags = 0U;
    batch_size_ = args[0]->shape[0];
    builder_->setMaxBatchSize(batch_size_);
  }
  network_ = builder_->createNetworkV2(flags);
#else
  // Use INetwork with implicit batch.
  batch_size_ = args[0]->shape[0];
  builder_->setMaxBatchSize(batch_size_);
  builder_->setMaxWorkspaceSize(max_workspace_size_);
  const bool use_fp16 = dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false);
  builder_->setFp16Mode(use_fp16);
  network_ = builder_->createNetwork();
#endif
}

void TensorRTBuilder::ProcessInputs(const Function& func) {
  // All input names in order. This order matches that of execution args.
  for (size_t i = 0; i < func->params.size(); i++) {
    network_input_names_.push_back(func->params[i]->name_hint());
    network_input_map_[func->params[i]->name_hint()] = i;
  }
  // Assume all inputs are real to start. If an input is baked into the TRT
  // engine, we will set the entry in this array to true.
  network_input_is_baked_.assign(func->params.size(), false);
}

void TensorRTBuilder::ProcessOutputs(const Expr& expr) {
  // Mark outputs.
  auto it = node_output_map_.find(expr.operator->());
  CHECK(it != node_output_map_.end()) << "Output was not found.";
  auto network_outputs = it->second;
  for (size_t i = 0; i < network_outputs.size(); ++i) {
    CHECK(network_outputs[i].type == kTensor);
    auto out_tensor = network_outputs[i].tensor;
    std::string output_name = "tensorrt_output" + std::to_string(i);
    // If the network is already marked as an output, make a copy to avoid TRT crash. This shouldn't
    // happen since duplicate output issue in partitioning was fixed.
    if (out_tensor->isNetworkOutput()) {
      LOG(WARNING) << output_name << " is a duplicate output.";
      out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
    } else if (out_tensor->isNetworkInput()) {
      LOG(WARNING) << output_name << " is also an input.";
      out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
    }
    out_tensor->setName(output_name.c_str());
    network_output_names_.push_back(output_name);
    network_->markOutput(*out_tensor);
    DLOG(INFO) << "Added TRT network output: " << out_tensor->getName()
               << " -> " << output_name;
  }
}

runtime::TrtEngineAndContext TensorRTBuilder::BuildEngine(
    const Function& func) {
  // Process graph to create INetworkDefinition.
  ProcessInputs(func);
  VisitExpr(func->body);
  ProcessOutputs(func->body);
// Build engine.
#if TRT_VERSION_GE(6, 0, 1)
  config_ = builder_->createBuilderConfig();
  config_->setMaxWorkspaceSize(max_workspace_size_);
  if (dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false)) {
    config_->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  // Add profiles.
  if (!use_implicit_batch_) {
    auto profile = builder_->createOptimizationProfile();
    for (int i = 0; i < network_->getNbInputs(); ++i) {
      auto name = network_->getInput(i)->getName();
      auto dims = network_->getInput(i)->getDimensions();
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, dims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, dims);
      profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, dims);
    }
    config_->addOptimizationProfile(profile);
  }
  nvinfer1::ICudaEngine* engine =
      builder_->buildEngineWithConfig(*network_, *config_);
#else
  nvinfer1::ICudaEngine* engine = builder_->buildCudaEngine(*network_);
#endif
  CleanUp();
  const int num_input_bindings = std::count(
      network_input_is_baked_.begin(), network_input_is_baked_.end(), false);
  CHECK_EQ(engine->getNbBindings(),
           num_input_bindings + network_output_names_.size());
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  return {engine, context, network_input_names_, network_input_is_baked_,
          network_output_names_};
}

nvinfer1::Weights TensorRTBuilder::GetDLTensorAsWeights(
    DLTensor* dptr, DLDeviceType src_device) {
  CHECK_EQ(dptr->ctx.device_type, src_device);
  CHECK(static_cast<int>(dptr->dtype.code) == kDLFloat ||
        static_cast<int>(dptr->dtype.code) == kDLInt);
  const auto trt_dtype = static_cast<int>(dptr->dtype.code) == kDLFloat
                             ? nvinfer1::DataType::kFLOAT
                             : nvinfer1::DataType::kINT32;
  const size_t weight_bytes = runtime::GetDataSize(*dptr);
  nvinfer1::Weights weight{trt_dtype, nullptr, 0};
  size_t count = 1;
  for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
    count *= dptr->shape[i];
  }
  CHECK_EQ(count * 4, weight_bytes);
  weight.count = count;
  weight.values = new float[count];
  CHECK_EQ(
      TVMArrayCopyToBytes(dptr, const_cast<void*>(weight.values), weight_bytes),
      0)
      << TVMGetLastError();
  trt_weights_.push_back(weight);
  return weight;
}

nvinfer1::Weights TensorRTBuilder::GetNdArrayAsWeights(
    const runtime::NDArray& array, DLDeviceType src_device) {
  DLTensor* dptr = const_cast<DLTensor*>(array.operator->());
  return GetDLTensorAsWeights(dptr, src_device);
}

void TensorRTBuilder::GetInputAsWeights(const VarNode* node) {
  const int var_node_idx = network_input_map_[node->name_hint()];
  // This input will be baked into TensorRT engine using value from first invocation.
  network_input_is_baked_[var_node_idx] = true;
  nvinfer1::Weights weight =
      GetDLTensorAsWeights(execution_args_[var_node_idx], kDLGPU);
  node_output_map_[node] = {TrtOpInput(weight, GetShape(node->checked_type()))};
}

void TensorRTBuilder::GetConstantAsWeights(const ConstantNode* node) {
  auto weight = GetNdArrayAsWeights(node->data, kDLCPU);
  auto shape_long = node->data.Shape();
  std::vector<int> shape(shape_long.begin(), shape_long.end());
  node_output_map_[node] = {TrtOpInput(weight, shape)};
}

void TensorRTBuilder::GetInputAsTransposedWeights(const CallNode* transpose,
                                                  const VarNode* node) {
  GetInputAsWeights(node);
  CHECK_EQ(node_output_map_[node].size(), 1);
  const nvinfer1::Weights& original_weight = node_output_map_[node][0].weight;
  const auto& shape = node_output_map_[node][0].weight_shape;
  const float* original_values =
      static_cast<const float*>(original_weight.values);
  float* values = new float[original_weight.count];
  // Get order and new shape.
  const auto* attrs = transpose->attrs.as<TransposeAttrs>();
  std::vector<int> order(attrs->axes.size(), 0);
  std::vector<int> new_shape(attrs->axes.size(), 0);
  for (size_t i = 0; i < attrs->axes.size(); ++i) {
    const int axis = attrs->axes[i].as<IntImmNode>()->value;
    order[i] = axis;
    new_shape[i] = shape[axis];
  }
  // Perform transpose.
  if (order.size() == 4 && order[0] == 3 && order[1] == 2 && order[2] == 0 &&
      order[3] == 1) {
    const int output_strides[4] = {shape[1], 1, shape[0] * shape[1],
                                   shape[0] * shape[1] * shape[2]};
    TransposeWeights4D(shape, output_strides, original_values, values);
  } else if (order.size() == 4 && order[0] == 2 && order[1] == 3 &&
             order[2] == 0 && order[3] == 1) {
    const int output_strides[4] = {shape[1], 1, shape[0] * shape[1] * shape[3],
                                   shape[0] * shape[1]};
    TransposeWeights4D(shape, output_strides, original_values, values);
  } else if (order.size() == 2 && order[0] == 1 && order[1] == 0) {
    TransposeWeights2D(shape, original_values, values);
  } else {
    LOG(FATAL) << "Constant transpose " << DebugString(order)
               << " is not supported.";
  }
  // Map as output of transpose op.
  nvinfer1::Weights transposed_weight{nvinfer1::DataType::kFLOAT, values,
                                      original_weight.count};
  trt_weights_.push_back(transposed_weight);
  node_output_map_[transpose] = {TrtOpInput(transposed_weight, new_shape)};
}

void TensorRTBuilder::VisitExpr_(const TupleGetItemNode* op) {
  if (const auto* tuple = op->tuple.as<TupleNode>()) {
    Expr item = tuple->fields[op->index];
    VisitExpr(item);
    node_output_map_[op] = node_output_map_[item.operator->()];
  } else {
    VisitExpr(op->tuple);
    // Index into tensor outputs from expr.
    node_output_map_[op] = {
        node_output_map_[op->tuple.operator->()][op->index]};
  }
}

void TensorRTBuilder::VisitExpr_(const TupleNode* op) {
  std::vector<TrtOpInput> outputs;
  for (auto item : op->fields) {
    VisitExpr(item);
    auto item_outputs = node_output_map_[item.operator->()];
    outputs.reserve(outputs.size() + item_outputs.size());
    outputs.insert(outputs.end(), item_outputs.begin(), item_outputs.end());
  }
  node_output_map_[op] = outputs;
}

nvinfer1::ITensor* TensorRTBuilder::AddInput(const std::string& tensor_name, const Type& type) {
  auto shape = GetShape(type);
  // Remove batch dim when not in explicit batch mode.
  if (use_implicit_batch_ && shape.size() > 1) {
    shape.erase(shape.begin());
  }
  DLOG(INFO) << "Added TRT network input: " << tensor_name << " "
             << DebugString(shape);
  nvinfer1::Dims dims = VectorToTrtDims(shape);
  auto type_node = type.as<TensorTypeNode>();
  CHECK(type_node != nullptr &&
        runtime::TypeMatch(type_node->dtype, kDLFloat, 32))
      << "Only FP32 inputs are supported.";
  return network_->addInput(tensor_name.c_str(), nvinfer1::DataType::kFLOAT, dims);
}

void TensorRTBuilder::VisitExpr_(const VarNode* node) {
  if (node->checked_type().as<TupleTypeNode>()) {
    // Handle TupleTypes by creating multiple TRT inputs from one.
    auto* tuple_type = node->type_as<TupleTypeNode>();
    std::vector<TrtOpInput> outputs;
    const std::string& original_name = node->name_hint();
    std::vector<std::string> new_names;
    for (int i = 0; i < tuple_type->fields.size(); ++i) {
      std::string tensor_name = original_name + "_" + std::to_string(i);
      new_names.push_back(tensor_name);
      outputs.push_back(TrtOpInput(AddInput(tensor_name, tuple_type->fields[i])));
    }
    node_output_map_[node] = outputs;
    // Update network_input_map_
    const int original_index = network_input_map_[original_name];
    network_input_map_.erase(original_name);
    // Push all other inputs back.
    for (auto it : network_input_map_) {
      if (it.second > original_index) {
        network_input_map_[it.first] += new_names.size() - 1;
      }
    }
    for (size_t i = 0; i < new_names.size(); ++i) {
      network_input_map_[new_names[i]] = original_index + i;
    }
    // Update network_input_names_
    network_input_names_.erase(network_input_names_.begin() + original_index);
    network_input_names_.insert(network_input_names_.begin() + original_index,
                                new_names.begin(), new_names.end());
    // Update network_input_is_baked_
    bool is_baked = network_input_is_baked_[original_index];
    network_input_is_baked_.erase(network_input_is_baked_.begin() + original_index);
    network_input_is_baked_.insert(network_input_is_baked_.begin() + original_index,
                                   new_names.size(), is_baked);
  } else if (node->checked_type().as<TensorTypeNode>()) {
    // Standard TensorType case.
    const std::string& tensor_name = node->name_hint();
    node_output_map_[node] = {TrtOpInput(AddInput(tensor_name, node->checked_type()))};
  } else {
    LOG(FATAL) << "VarNode must be Tensor or Tuple type.";
  }
}

void TensorRTBuilder::VisitExpr_(const ConstantNode* node) {
  nvinfer1::Weights weight = GetNdArrayAsWeights(node->data, kDLCPU);
  auto shape = node->data.Shape();
  // Remove batch dim when not in explicit batch mode.
  if (use_implicit_batch_ && shape.size() > 1 && shape[0] == 1) {
    shape.erase(shape.begin());
  }
  nvinfer1::Dims dims = VectorToTrtDims(shape);
  auto const_layer = network_->addConstant(dims, weight);
  CHECK(const_layer != nullptr);
  node_output_map_[node] = {TrtOpInput(const_layer->getOutput(0))};
}

void TensorRTBuilder::VisitExpr_(const CallNode* call) {
  AddTrtLayerParams params(network_, call, &trt_weights_);
  // Look up converter.
  auto it = GetOpConverters()->find(params.op_name);
  CHECK(it != GetOpConverters()->end())
      << "Unsupported operator conversion to TRT, op name: " << params.op_name;
  const auto converter = it->second;

  // Ensure that nodes are processed in topological order by visiting their inputs first.
  for (size_t i = 0; i < call->args.size(); ++i) {
    if (converter->variable_input_count ||
        converter->input_types[i] != kWeight) {
      VisitExpr(call->args[i]);
      continue;
    }
    // Handle special case where input must be constant array on CPU.
    if (auto* var = call->args[i].as<VarNode>()) {
      GetInputAsWeights(var);
    } else if (auto* node = call->args[i].as<ConstantNode>()) {
      GetConstantAsWeights(node);
    } else {
      // Temporary workaround for transposed weights. Once partitioning is
      // available, the transpose will be computed by tvm and the result will be
      // a var input. Also not needed when params are bound to constants since
      // FoldConstants will remove the transpose for us.
      const CallNode* transpose = call->args[i].as<CallNode>();
      const VarNode* weights = nullptr;
      if (transpose && transpose->op.as<OpNode>()->name == "transpose" &&
          (weights = transpose->args[0].as<VarNode>())) {
        GetInputAsTransposedWeights(transpose, weights);
      } else {
        LOG(FATAL) << "TRT requires a constant input here.";
      }
    }
  }

  // Get inputs.
  for (size_t i = 0; i < call->args.size(); ++i) {
    auto it = node_output_map_.find(call->args[i].operator->());
    CHECK(it != node_output_map_.end()) << "Input was not found.";
    for (auto out : it->second) {
      params.inputs.push_back(out);
    }
  }
  if (!converter->variable_input_count) {
    CHECK_EQ(converter->input_types.size(), params.inputs.size())
        << "Op expected a different number of inputs.";
  }

  // Convert op to TRT.
  converter->Convert(&params);

  // Get outputs.
  node_output_map_[call] = {};
  std::vector<TrtOpInput> outputs;
  for (auto out : params.outputs) {
    node_output_map_[call].push_back(TrtOpInput(out));
  }
}

void TensorRTBuilder::CleanUp() {
  network_->destroy();
#if TRT_VERSION_GE(6, 0, 1)
  config_->destroy();
#endif
  builder_->destroy();
  for (auto weight : trt_weights_) {
    if (weight.type == nvinfer1::DataType::kFLOAT) {
      delete[] static_cast<const float*>(weight.values);
    } else {
      delete[] static_cast<const uint16_t*>(weight.values);
    }
  }
}

void TransposeWeights4D(const std::vector<int>& original_shape,
                        const int* output_strides, const float* input_values,
                        float* output_values) {
  const int input_strides[4] = {
      original_shape[1] * original_shape[2] * original_shape[3],
      original_shape[2] * original_shape[3], original_shape[3], 1};
  for (int i = 0; i < original_shape[0]; i++) {
    for (int j = 0; j < original_shape[1]; j++) {
      for (int k = 0; k < original_shape[2]; k++) {
        for (int l = 0; l < original_shape[3]; l++) {
          const int input_index =
              (i * input_strides[0]) + (j * input_strides[1]) +
              (k * input_strides[2]) + (l * input_strides[3]);
          const int output_index =
              (i * output_strides[0]) + (j * output_strides[1]) +
              (k * output_strides[2]) + (l * output_strides[3]);
          output_values[output_index] = input_values[input_index];
        }
      }
    }
  }
}

void TransposeWeights2D(const std::vector<int>& original_shape,
                        const float* input_values, float* output_values) {
  const int c = original_shape[0];
  const int k = original_shape[1];
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < k; j++) {
      const int input_index = i * k + j;
      const int output_index = j * c + i;
      output_values[output_index] = input_values[input_index];
    }
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
