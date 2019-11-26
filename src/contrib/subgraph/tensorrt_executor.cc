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

#include <tvm/runtime/c_runtime_api.h>
#include <dmlc/parameter.h>
#include <dmlc/timer.h>
#include <unordered_set>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include "./subgraph.h"
#include "./tensorrt_executor.h"
#include "../../runtime/cuda/cuda_common.h"

namespace tvm {
namespace contrib {

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if (res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", \
      #x, res, cudaGetErrorString(res), __FILE__, __LINE__); \
    exit(1); \
  } \
} while (0)

static size_t GetTensorSize(const DLTensor& arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr.ndim; ++i) {
    size *= arr.shape[i];
  }
  return size;
}

static size_t GetTensorBytes(const DLTensor& arr) {
  size_t size = GetTensorSize(arr);
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

// Logger for TensorRT info/warning/errors
class TensorRTLogger : public nvinfer1::ILogger {
 public:
  TensorRTLogger(): TensorRTLogger(Severity::kWARNING) {}
  explicit TensorRTLogger(Severity severity): reportable_severity(severity) {}
  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR: LOG(ERROR) << "INTERNAL_ERROR: " << msg; break;
      case Severity::kERROR: LOG(ERROR) << "ERROR: " << msg; break;
      case Severity::kWARNING: LOG(WARNING) << "WARNING: " << msg; break;
      case Severity::kINFO: LOG(INFO) << "INFO: " << msg; break;
      default: LOG(INFO) << "UNKNOWN: " << msg; break;
    }
  }
 private:
  Severity reportable_severity{Severity::kWARNING};
};

class TensorRTProfiler : public nvinfer1::IProfiler {
 public:
  explicit TensorRTProfiler(const std::string& subgraph_name) :
      subgraph_name_(subgraph_name) {}

  virtual void reportLayerTime(const char* layer_name, float milli_sec) {
    auto it = record_map_.find(layer_name);
    if (it != record_map_.end()) {
      it->second += milli_sec;
    } else {
      record_map_.emplace(layer_name, milli_sec);
      layer_names_.push_back(layer_name);
    }
  }

  void PrintSummary() {
    float total_time = 0;
    for (const auto& layer_name : layer_names_) {
      auto it = record_map_.find(layer_name);
      CHECK(it != record_map_.end());
      LOG(INFO) << "TensorRT subgraph: " << subgraph_name_ << ", layer: " << layer_name
                << ", time cost: " << it->second << "ms";
      total_time += it->second;
    }
    LOG(INFO) << "TensorRT subgraph: " << subgraph_name_ << ", total time cost: "
              << total_time << "ms";
  }

 private:
  std::string subgraph_name_;
  std::vector<std::string> layer_names_;
  std::unordered_map<std::string, float> record_map_;
};

TensorRTExecManager::TensorRTExecManager() {
  static TensorRTLogger trt_logger;
  infer_engine_builder_ = nvinfer1::createInferBuilder(trt_logger);
  max_workspace_size_ = dmlc::GetEnv("TVM_TENSORRT_MAX_WORKSPACE_SIZE", 1 << 29);
  use_fp16_ = dmlc::GetEnv("TVM_TENSORRT_USE_FP16", false);
  use_profiler_ = dmlc::GetEnv("TVM_TENSORRT_USE_PROFILER", false);
}

TensorRTExecManager::~TensorRTExecManager() {
  for (auto kv : infer_engine_context_map_) {
    // DO NOT change the following two lines' order
    kv.second->destroy();  // destroy context
    kv.first->destroy();  // destroy engine
  }
  if (infer_engine_builder_ != nullptr) {
    infer_engine_builder_->destroy();
  }
}

// data_entries contains inputs and outputs of the subgraph in topo-sorted order
std::function<void()> TensorRTExecManager::CreateExec(const std::string& subgraph_name,
                                                      const Subgraph& subgraph,
                                                      const std::vector<DLTensor>& data_entries) {
  CHECK_EQ(infer_engine_map_.count(subgraph_name), 0U);
  CHECK_EQ(subgraph.arg_nodes.size() + subgraph.heads.size(), data_entries.size());
  auto exec = [this, &subgraph_name, &subgraph, data_entries] () {
    auto it = this->infer_engine_map_.find(subgraph_name);
    if (it == this->infer_engine_map_.end()) {
      // input_data_idx stores the indices of input data in data_entries
      std::vector<uint32_t> input_data_idx;
      // input_data_names stores the input data node name in the order of input_data_idx
      std::vector<std::string> input_data_names;
      std::vector<std::string> output_names;
      nvinfer1::ICudaEngine* engine = CreateInferEngine(
          subgraph, data_entries, &input_data_idx, &input_data_names, &output_names);
      CHECK_EQ(input_data_names.size(), input_data_idx.size());
      const int num_bindings = engine->getNbBindings();
      CHECK_EQ(static_cast<size_t>(num_bindings), input_data_idx.size() + output_names.size());
      this->infer_engine_map_.emplace(subgraph_name, engine);
      nvinfer1::IExecutionContext* context = engine->createExecutionContext();
      CHECK(context != nullptr);
      this->infer_engine_context_map_.emplace(engine, context);
      this->input_data_idx_map_.emplace(engine, input_data_idx);
      this->input_data_name_map_.emplace(engine, input_data_names);
      this->output_name_map_.emplace(engine, output_names);
    } else if (infer_engine_builder_ != nullptr) {
      infer_engine_builder_->destroy();
      infer_engine_builder_ = nullptr;
    }
    nvinfer1::ICudaEngine* engine = this->infer_engine_map_.at(subgraph_name);
    nvinfer1::IExecutionContext* context = this->infer_engine_context_map_.at(engine);
    const std::vector<uint32_t>& input_data_idx = this->input_data_idx_map_.at(engine);
    const std::vector<std::string>& input_data_names = this->input_data_name_map_.at(engine);
    const std::vector<std::string>& output_names = this->output_name_map_.at(engine);
    // total num of inputs and outputs (excluding weights)
    const int num_bindings = engine->getNbBindings();
    void* bindings[num_bindings];
    for (size_t i = 0; i < input_data_idx.size(); ++i) {
      const int input_idx = engine->getBindingIndex(input_data_names[i].c_str());
      DLTensor tensor = data_entries[input_data_idx[i]];
      bindings[input_idx] = static_cast<char*>(tensor.data) + tensor.byte_offset;
    }
    for (size_t i = subgraph.arg_nodes.size(); i < data_entries.size(); ++i) {
      DLTensor tensor = data_entries[i];
      const int output_idx = engine->getBindingIndex(
          output_names[i-subgraph.arg_nodes.size()].c_str());
      bindings[output_idx] = static_cast<char*>(tensor.data) + tensor.byte_offset;
    }
    const int batch_size = data_entries[input_data_idx[0]].shape[0];
    // const double start = dmlc::GetTime();
    TensorRTProfiler profiler(subgraph_name);
    if (use_profiler_) {
      context->setProfiler(&profiler);
    }
    CHECK(context->execute(batch_size, bindings)) << "Running TensorRT for subgraph "
      << subgraph_name << " failed.";
    if (use_profiler_) {
      profiler.PrintSummary();
    }
    // context->enqueue(batch_size, bindings,
    //     runtime::CUDAThreadEntry::ThreadLocal()->stream, nullptr);
    // CHECK_CUDART(cudaStreamSynchronize(runtime::CUDAThreadEntry::ThreadLocal()->stream));
    // LOG(INFO) << "TensorRT inference time: " << (dmlc::GetTime() - start) * 1000. << "ms";
  };
  return exec;
}

/*!
 * Gets or creates an nvinfer1::Weights based upon the subgraph weight.
 * allocated on GPU.
 * @param nid Weight data node id in the subgraph
 * @param data_entries Inputs and outputs of the subgraph
 * @param input_nid2idx Mapping of node id to its index in data_entries
 * @param nid2weights Mapping of node id to its TensorRT Weights
 * @return nvinfer1::Weights allocated on CPU
 */
nvinfer1::Weights GetTensorRTWeights(
    const uint32_t nid,
    const std::vector<DLTensor>& data_entries,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights) {
  CHECK(input_nid2idx.count(nid))
    << "Weight of conv2d must come from an input of the subgraph";
  if (!nid2weights->count(nid)) {
    // param has not been created
    DLTensor weight = data_entries[input_nid2idx.at(nid)];
    CHECK_EQ(weight.ctx.device_type, kDLGPU);
    CHECK_EQ(static_cast<int>(weight.dtype.code), kDLFloat);
    const size_t weight_bytes = GetTensorBytes(weight);
    nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
    wt.values = malloc(weight_bytes);
    wt.count = GetTensorSize(weight);
    CHECK_EQ(TVMArrayCopyToBytes(&weight, const_cast<void*>(wt.values), weight_bytes), 0)
        << TVMGetLastError();
    nid2weights->emplace(nid, wt);
  }
  return nid2weights->at(nid);
}

/*!
 * Gets or creates an input nvinfer1::ITensor for the operator with node_name.
 * @param node_name Operator node name whose input tensor is going to be returned
 * @param nodes All the nodes in the subgraph
 * @param input_data_entry Input NodeEntry of the operator with node_name
 * @param data_entries Inputs and outputs of the subgraph
 * @param input_nid2idx Mapping of node id to its index in data_entries
 * @param nid2layer Mapping node id to its corresponding TensorRT layer
 * @param network The TensorRT network representing the subgraph
 * @param nid2tensor Mapping from node id to its TensorRT tensor
 * @param input_data_idx Input data indices of data_entries
 * @param input_data_names Input data names of data_entries
 * @return
 */
nvinfer1::ITensor* GetTensorRTTensor(
    const std::string& node_name,  // node that takes input_data_entry as input
    const std::vector<Subgraph::Node>& nodes,
    const Subgraph::Node::NodeEntry& input_data_entry,
    const std::vector<DLTensor>& data_entries,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::unordered_map<uint32_t, nvinfer1::ILayer*>& nid2layer,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names) {
  nvinfer1::ITensor* data = nullptr;
  if (input_nid2idx.count(input_data_entry.node_id)) {
    // if the input data is from an input of the subgraph
    CHECK_EQ(input_data_entry.index, 0U);
    if (!nid2tensor->count(input_data_entry.node_id)) {
      // if data has not been created, create one
      const DLTensor& tensor = data_entries[input_nid2idx.at(input_data_entry.node_id)];
      CHECK_EQ(static_cast<int>(tensor.dtype.code), kDLFloat);
      CHECK(tensor.ndim >= 2 && tensor.ndim <= 4) << "Unsupported tensor ndim = " << tensor.ndim
                                                  << " for node " << node_name;
      nvinfer1::Dims dims;
      // Note: The batch size is set to 1 for building the network.
      // The real batch size is set through execute or enqueue functions.
      if (tensor.ndim == 2) {
        dims = nvinfer1::Dims2(1, tensor.shape[1]);
        dims.type[0] = nvinfer1::DimensionType::kINDEX;
        dims.type[1] = nvinfer1::DimensionType::kSPATIAL;
      } else if (tensor.ndim == 3) {
        dims = nvinfer1::Dims3(1, tensor.shape[1], tensor.shape[2]);
        dims.type[0] = nvinfer1::DimensionType::kINDEX;
        dims.type[1] = nvinfer1::DimensionType::kCHANNEL;
        dims.type[2] = nvinfer1::DimensionType::kSPATIAL;
      } else if (tensor.ndim == 4) {
        dims = nvinfer1::DimsNCHW(1, tensor.shape[1],
                                  tensor.shape[2], tensor.shape[3]);
      }
      input_data_names->push_back(nodes[input_data_entry.node_id].node_name);
      input_data_idx->push_back(input_nid2idx.at(input_data_entry.node_id));
      nid2tensor->emplace(input_data_entry.node_id, network->addInput(
          input_data_names->back().c_str(),
          nvinfer1::DataType::kFLOAT, dims));
    }
    data = nid2tensor->at(input_data_entry.node_id);
    data->setName(nodes[input_data_entry.node_id].node_name.c_str());
  } else {
    // the input data is provided by a layer's output
    auto it = nid2layer.find(input_data_entry.node_id);
    CHECK(it != nid2layer.end())
      << node_name << " cannot depends on the output of layer"
      << nodes[input_data_entry.node_id].node_name
      << " that is executed after it";
    const std::string layer_output_name = std::string(it->second->getName())
                                        + "_output" + std::to_string(input_data_entry.index);
    data = it->second->getOutput(input_data_entry.index);
    data->setName(layer_output_name.c_str());
  }
  CHECK(data != nullptr);
  return data;
}

std::vector<std::string> TokenizeTuple(const std::string& tuple) {
  CHECK(tuple.front() == '(' || tuple.front() == '[');
  CHECK(tuple.back() == ')' || tuple.back() == ']');
  std::stringstream ss(tuple.substr(1, tuple.size() - 2U));
  std::vector<std::string> ret;
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, ',');
    ret.push_back(substr);
  }
  CHECK(!ret.empty()) << "Tuple " << tuple << " contains no data";
  return ret;
}

void AddConvolution(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  // create or get conv2d input data which is an ITensor*
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  // create conv2d weight and bias on cpu since TensorRT API only accepts that
  nvinfer1::Weights weight = GetTensorRTWeights(
      nodes[nid].inputs[1].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  if (nodes[nid].inputs.size() == 3U) {
    CHECK(!nodes[nid].attrs.count("use_bias") || nodes[nid].attrs.at("use_bias") == "True"
          || nodes[nid].attrs.at("use_bias") == "1");
    bias = GetTensorRTWeights(
        nodes[nid].inputs[2].node_id, data_entries, input_nid2idx, nid2weights);
  } else {
    CHECK(nodes[nid].attrs.count("use_bias") && (nodes[nid].attrs.at("use_bias") == "False"
          || nodes[nid].attrs.at("use_bias") == "0"));
  }
  CHECK(!nodes[nid].attrs.count("layout") || nodes[nid].attrs.at("layout") == "NCHW");
  std::vector<std::string> tokens = TokenizeTuple(nodes[nid].attrs.at("kernel_size"));
  CHECK_EQ(tokens.size(), 2U);
  nvinfer1::IConvolutionLayer* conv_layer = network->addConvolution(
      *data, std::stoi(nodes[nid].attrs.at("channels")),
      nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])),
      weight, bias);
  CHECK(conv_layer != nullptr);
  conv_layer->setName(nodes[nid].node_name.c_str());
  if (nodes[nid].attrs.count("padding")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("padding"));
    CHECK_EQ(tokens.size(), 2U);
    conv_layer->setPadding(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  if (nodes[nid].attrs.count("strides")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("strides"));
    CHECK_EQ(tokens.size(), 2U);
    conv_layer->setStride(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  if (nodes[nid].attrs.count("groups")) {
    conv_layer->setNbGroups(std::stoi(nodes[nid].attrs.at("groups")));
  }
  nid2layer->emplace(nid, conv_layer);
}

void AddBatchNorm(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  CHECK_EQ(nodes[nid].inputs.size(), 5U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  // get batch norm gamma, beta, moving_mean, moving_var on cpu
  nvinfer1::Weights gamma = GetTensorRTWeights(
      nodes[nid].inputs[1].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights beta = GetTensorRTWeights(
      nodes[nid].inputs[2].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights mean = GetTensorRTWeights(
      nodes[nid].inputs[3].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights var = GetTensorRTWeights(
      nodes[nid].inputs[4].node_id, data_entries, input_nid2idx, nid2weights);
  CHECK_EQ(gamma.count, beta.count);
  CHECK_EQ(gamma.count, mean.count);
  CHECK_EQ(gamma.count, var.count);
  const int axis = nodes[nid].attrs.count("axis")? std::stoi(nodes[nid].attrs.at("axis")) : 1;
  CHECK_EQ(axis, 1) << "Only support channel axis = 1";
  // TODO(junwu): implement a parser for parsing attrs
  const float epsilon = nodes[nid].attrs.count("epsilon")?
      std::stof(nodes[nid].attrs.at("epsilon")) : 1e-5;
  const bool center = nodes[nid].attrs.count("center")?
      nodes[nid].attrs.at("center") == "True" : true;
  const bool scale = nodes[nid].attrs.count("scale")? nodes[nid].attrs.at("scale") == "True" : 1;

  // We want to convert batch_norm equation to using IScaleLayer in TensorRT.
  // In this case, the scale of the IScaleLayer is gamma/sqrt(var+epsilon),
  // and shift is beta - mean/scale.
  void* weight_scale_ptr = malloc(sizeof(float) * gamma.count);
  nvinfer1::Weights weight_scale{nvinfer1::DataType::kFLOAT, weight_scale_ptr, gamma.count};
  extra_weights->push_back(weight_scale);
  void* weight_shift_ptr = malloc(sizeof(float) * gamma.count);
  nvinfer1::Weights weight_shift{nvinfer1::DataType::kFLOAT, weight_shift_ptr, gamma.count};
  extra_weights->push_back(weight_shift);
  nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};

  // fill in the content of weights for the Scale layer
  const float* gamma_ptr = reinterpret_cast<const float*>(gamma.values);
  const float* beta_ptr = reinterpret_cast<const float*>(beta.values);
  const float* mean_ptr = reinterpret_cast<const float*>(mean.values);
  const float* var_ptr = reinterpret_cast<const float*>(var.values);
  float* scale_ptr = reinterpret_cast<float*>(weight_scale_ptr);
  float* shift_ptr = reinterpret_cast<float*>(weight_shift_ptr);
  // TODO(junwu): consider parallelizing the following loop
  for (int i = 0; i < gamma.count; ++i) {
    scale_ptr[i] = 1.0 / std::sqrt(var_ptr[i] + epsilon);
    if (scale) {
      scale_ptr[i] *= gamma_ptr[i];
    }
    shift_ptr[i] = - mean_ptr[i] * scale_ptr[i];
    if (center) {
      shift_ptr[i] += beta_ptr[i];
    }
  }
  nvinfer1::IScaleLayer* scale_layer = network->addScale(
      *data, nvinfer1::ScaleMode::kCHANNEL, weight_shift, weight_scale, power);
  CHECK(scale_layer != nullptr);
  scale_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, scale_layer);
}

void AddActivation(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  static const std::unordered_map<std::string, nvinfer1::ActivationType> op_map =
      {{"relu", nvinfer1::ActivationType::kRELU},
       {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
       {"tanh", nvinfer1::ActivationType::kTANH}};
  auto it = op_map.find(nodes[nid].op_name);
  CHECK(it != op_map.end()) << "Unsupported activation type "
                            << nodes[nid].op_name << " in TensorRT";
  nvinfer1::IActivationLayer* act_layer = network->addActivation(*data, it->second);
  CHECK(act_layer != nullptr);
  act_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, act_layer);
}

void AddElementWiseBinaryOp(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation> op_map =
      {{"elemwise_add", nvinfer1::ElementWiseOperation::kSUM},
       {"elemwise_sub", nvinfer1::ElementWiseOperation::kSUB},
       {"elemwise_mul", nvinfer1::ElementWiseOperation::kPROD},
       {"elemwise_div", nvinfer1::ElementWiseOperation::kDIV},
       {"elemwise_pow", nvinfer1::ElementWiseOperation::kPOW}};
  CHECK_EQ(nid2layer->count(nid), 0U);
  CHECK_EQ(nodes[nid].inputs.size(), 2U);
  nvinfer1::ITensor* data[2];
  for (int i = 0; i < 2; ++i) {
    data[i] = GetTensorRTTensor(
        nodes[nid].node_name, nodes, nodes[nid].inputs[i], data_entries,
        input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  }
  auto it = op_map.find(nodes[nid].op_name);
  CHECK(it != op_map.end()) << "Unsupported element-wise binary op "
                            << nodes[nid].op_name << " in TensorRT";
  nvinfer1::IElementWiseLayer* elemwise_layer =
      network->addElementWise(*(data[0]), *(data[1]), it->second);
  CHECK(elemwise_layer != nullptr);
  elemwise_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, elemwise_layer);
}

void AddPooling(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  static const std::unordered_map<std::string, nvinfer1::PoolingType> op_map =
      {{"max_pool2d", nvinfer1::PoolingType::kMAX},
       {"avg_pool2d", nvinfer1::PoolingType::kAVERAGE},
       {"global_max_pool2d", nvinfer1::PoolingType::kMAX},
       {"global_avg_pool2d", nvinfer1::PoolingType::kAVERAGE}};
  auto it = op_map.find(nodes[nid].op_name);
  CHECK(it != op_map.end()) << "Unsupported pooling type " << nodes[nid].op_name << " in TensorRT";
  CHECK(!nodes[nid].attrs.count("layout") || nodes[nid].attrs.at("layout") == "NCHW");
  const bool is_global_pool = it->first.find("global") != std::string::npos;
  std::vector<std::string> tokens;
  nvinfer1::IPoolingLayer* pool_layer = nullptr;
  if (is_global_pool) {
    const nvinfer1::Dims data_dims = data->getDimensions();
    CHECK_EQ(data_dims.nbDims, 4);
    const int* data_shape = data_dims.d;
    pool_layer = network->addPooling(
        *data, it->second, nvinfer1::DimsHW(data_shape[2], data_shape[3]));
  } else {
    tokens = TokenizeTuple(nodes[nid].attrs.at("pool_size"));
    CHECK_EQ(tokens.size(), 2U);
    pool_layer = network->addPooling(
        *data, it->second, nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  CHECK(pool_layer != nullptr);
  pool_layer->setName(nodes[nid].node_name.c_str());
  if (nodes[nid].attrs.count("padding")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("padding"));
    if (tokens.size() == 1U) {
      pool_layer->setPadding(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[0])));
    } else if (tokens.size() == 2U) {
      pool_layer->setPadding(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
    } else if (tokens.size() == 4U) {
      CHECK_EQ(tokens[0], tokens[2]);
      CHECK_EQ(tokens[1], tokens[3]);
      pool_layer->setPadding(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
    } else {
      LOG(FATAL) << "Unsupported padding in TensorRT " << nodes[nid].attrs.at("padding");
    }
  }
  if (nodes[nid].attrs.count("strides")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("strides"));
    CHECK_EQ(tokens.size(), 2U);
    pool_layer->setStride(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  if (!is_global_pool && (nodes[nid].attrs.count("ceil_mode")
      && nodes[nid].attrs.at("ceil_mode") == "True")) {
    LOG(WARNING) << "Node " << nodes[nid].node_name << " used ceil mode for operator "
                 << nodes[nid].op_name
                 << ", which is currently not supported.";
  } else {
    network->setPoolingOutputDimensionsFormula(nullptr);
  }
  if (!is_global_pool) {
    if (nodes[nid].attrs.count("count_include_pad")) {
      if (nodes[nid].attrs.at("count_include_pad") == "True") {
        pool_layer->setAverageCountExcludesPadding(false);
      } else {
       pool_layer->setAverageCountExcludesPadding(true);
      }
    } else {
      pool_layer->setAverageCountExcludesPadding(true);
    }
  }
  nid2layer->emplace(nid, pool_layer);
}

void AddFullyConnected(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  nvinfer1::Weights weight = GetTensorRTWeights(
      nodes[nid].inputs[1].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  if (!nodes[nid].attrs.count("use_bias") || nodes[nid].attrs.at("use_bias") == "True"
      || nodes[nid].attrs.at("use_bias") == "1") {
    CHECK_EQ(nodes[nid].inputs.size(), 3U);
    bias = GetTensorRTWeights(
        nodes[nid].inputs[2].node_id, data_entries, input_nid2idx, nid2weights);
  } else {
    CHECK_EQ(nodes[nid].inputs.size(), 2U);
  }
  const int units = std::stoi(nodes[nid].attrs.at("units"));
  // TODO(junwu): The following code of changing input data dims is hacky. We need this
  // because TensorRT FC layer cannot accept dims less than 3. We should ask Nvidia to
  // remove this restriction in TensorRT in the future.
  nvinfer1::Dims data_dims = data->getDimensions();
  CHECK_GE(data_dims.nbDims, 2);
  if (data_dims.nbDims == 2) {
    LOG(INFO) << "Tensor " << data->getName() << " is an input of FullyConnected layer "
                                              << nodes[nid].node_name << " in TensorRT."
                                                 " Its ndim is 2, while TensorRT can only accept"
                                                 " ndim >= 3. Reset ndim to 4 by expanding two"
                                                 " trailing dims with dim size = 1.";
    nvinfer1::DimsNCHW dims_4d(data_dims.d[0], data_dims.d[1], 1, 1);
    data->setDimensions(dims_4d);
  }
  nvinfer1::IFullyConnectedLayer* fc_layer = network->addFullyConnected(*data, units, weight, bias);
  CHECK(fc_layer != nullptr);
  fc_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, fc_layer);
}

void AddSoftmax(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  // CHECK(!nodes[nid].attrs.count("axis"));
  nvinfer1::Dims data_dims = data->getDimensions();
  nvinfer1::ISoftMaxLayer* softmax_layer = network->addSoftMax(*data);
  CHECK(softmax_layer != nullptr);
  softmax_layer->setName(nodes[nid].node_name.c_str());
  if (data_dims.nbDims == 2) {
    softmax_layer->setAxes(2);
  }
  nid2layer->emplace(nid, softmax_layer);
}

void AddConcatenate(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  const int num_inputs = nodes[nid].inputs.size();
  std::vector<nvinfer1::ITensor*> input_tensors(num_inputs, nullptr);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = GetTensorRTTensor(
        nodes[nid].node_name, nodes, nodes[nid].inputs[i], data_entries,
        input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  }
  const int axis = !nodes[nid].attrs.count("axis")? 1 : std::stoi(nodes[nid].attrs.at("axis"));
  CHECK_GE(axis, 0);
  nvinfer1::IConcatenationLayer* concat_layer =
      network->addConcatenation(input_tensors.data(), num_inputs);
  CHECK(concat_layer != nullptr);
  concat_layer->setAxis(axis);
  concat_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, concat_layer);
}

void AddDeconvolution(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  nvinfer1::ITensor* data = GetTensorRTTensor(
      nodes[nid].node_name, nodes, nodes[nid].inputs[0], data_entries,
      input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  nvinfer1::Weights weight = GetTensorRTWeights(
      nodes[nid].inputs[1].node_id, data_entries, input_nid2idx, nid2weights);
  nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, nullptr, 0};
  if (nodes[nid].inputs.size() == 3U) {
    CHECK(!nodes[nid].attrs.count("use_bias") || nodes[nid].attrs.at("use_bias") == "True");
    bias = GetTensorRTWeights(
        nodes[nid].inputs[2].node_id, data_entries, input_nid2idx, nid2weights);
  } else {
    CHECK(nodes[nid].attrs.count("use_bias") && nodes[nid].attrs.at("use_bias") == "False");
  }
  CHECK(!nodes[nid].attrs.count("layout") || nodes[nid].attrs.at("layout") == "NCHW");
  std::vector<std::string> tokens = TokenizeTuple(nodes[nid].attrs.at("kernel_size"));
  nvinfer1::IDeconvolutionLayer* deconv_layer = network->addDeconvolution(
      *data, std::stoi(nodes[nid].attrs.at("channels")),
      nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])),
      weight, bias);
  CHECK(deconv_layer != nullptr);
  deconv_layer->setName(nodes[nid].node_name.c_str());
  if (nodes[nid].attrs.count("padding")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("padding"));
    deconv_layer->setPadding(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  if (nodes[nid].attrs.count("strides")) {
    tokens = TokenizeTuple(nodes[nid].attrs.at("strides"));
    deconv_layer->setStride(nvinfer1::DimsHW(std::stoi(tokens[0]), std::stoi(tokens[1])));
  }
  if (nodes[nid].attrs.count("groups")) {
    deconv_layer->setNbGroups(std::stoi(nodes[nid].attrs.at("groups")));
  }
  nid2layer->emplace(nid, deconv_layer);
}

void AddSliceLike(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weights) {
  CHECK_EQ(nid2layer->count(nid), 0U);
  const int num_inputs = nodes[nid].inputs.size();
  CHECK_EQ(num_inputs, 2);
  std::vector<nvinfer1::ITensor*> input_tensors(num_inputs, nullptr);
  for (int i = 0; i < num_inputs; ++i) {
    input_tensors[i] = GetTensorRTTensor(
        nodes[nid].node_name, nodes, nodes[nid].inputs[i], data_entries,
        input_nid2idx, *nid2layer, network, nid2tensor, input_data_idx, input_data_names);
  }
  CHECK_EQ(nodes[nid].attrs.count("axis"), 1U);
  CHECK_EQ(nodes[nid].attrs.count("offset"), 1U);
  std::vector<std::string> axes = TokenizeTuple(nodes[nid].attrs.at("axis"));
  CHECK_EQ(std::stoi(axes[0]), 2);
  CHECK_EQ(std::stoi(axes[1]), 3);
  std::vector<std::string> offsets = TokenizeTuple(nodes[nid].attrs.at("offset"));
  CHECK_EQ(offsets.size(), 2U);
  nvinfer1::Dims src_shape = input_tensors[0]->getDimensions();
  nvinfer1::Dims target_shape = input_tensors[1]->getDimensions();
  CHECK_EQ(src_shape.nbDims, 4);
  CHECK_EQ(target_shape.nbDims, 4);
  CHECK_GE(src_shape.d[2], target_shape.d[2]);
  CHECK_GE(src_shape.d[3], target_shape.d[3]);
  nvinfer1::DimsHW pre_padding(-std::stoi(offsets[0]), -std::stoi(offsets[1]));
  CHECK_LE(pre_padding.d[0], 0);
  CHECK_LE(pre_padding.d[1], 0);
  nvinfer1::DimsHW post_padding(
      target_shape.d[2]-pre_padding.d[0]-src_shape.d[2],
      target_shape.d[3]-pre_padding.d[1]-src_shape.d[3]);
  CHECK_LE(post_padding.d[0], 0);
  CHECK_LE(post_padding.d[1], 0);
  nvinfer1::IPaddingLayer* padding_layer =
      network->addPadding(*input_tensors[0], pre_padding, post_padding);
  CHECK(padding_layer != nullptr);
  padding_layer->setName(nodes[nid].node_name.c_str());
  nid2layer->emplace(nid, padding_layer);
}

namespace {
using AddTensorRTLayer = std::function<void(
    const uint32_t nid,
    const std::vector<Subgraph::Node>& nodes,
    const std::unordered_map<uint32_t, size_t>& input_nid2idx,
    const std::vector<DLTensor>& data_entries,
    nvinfer1::INetworkDefinition* network,
    std::unordered_map<uint32_t, nvinfer1::Weights>* nid2weights,
    std::unordered_map<uint32_t, nvinfer1::ITensor*>* nid2tensor,
    std::unordered_map<uint32_t, nvinfer1::ILayer*>* nid2layer,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<nvinfer1::Weights>* extra_weight)>;

static const std::unordered_map<std::string, AddTensorRTLayer> add_trt_layer_funcs =
    {{"conv2d", AddConvolution},
     {"batch_norm", AddBatchNorm},
     {"relu", AddActivation},
     {"sigmoid", AddActivation},
     {"tanh", AddActivation},
     {"elemwise_add", AddElementWiseBinaryOp},
     {"elemwise_sub", AddElementWiseBinaryOp},
     {"elemwise_mul", AddElementWiseBinaryOp},
     {"elemwise_div", AddElementWiseBinaryOp},
     {"elemwise_pow", AddElementWiseBinaryOp},
     {"max_pool2d", AddPooling},
     {"avg_pool2d", AddPooling},
     {"global_max_pool2d", AddPooling},
     {"global_avg_pool2d", AddPooling},
     {"dense", AddFullyConnected},
     {"softmax", AddSoftmax},
     {"concatenate", AddConcatenate},
     {"conv2d_transpose", AddDeconvolution},
     {"slice_like", AddSliceLike}
    };
}  // namespace

nvinfer1::ICudaEngine* TensorRTExecManager::CreateInferEngine(
    const Subgraph& subgraph,
    const std::vector<DLTensor>& data_entries,
    std::vector<uint32_t>* input_data_idx,
    std::vector<std::string>* input_data_names,
    std::vector<std::string>* output_names) {
  // maps input node id to its idx in data_entries
  std::unordered_map<uint32_t, size_t> input_nid2idx;
  for (size_t i = 0; i < subgraph.arg_nodes.size(); ++i) {
    input_nid2idx[subgraph.arg_nodes[i]] = i;
  }
  // maps nid to Weights
  std::unordered_map<uint32_t, nvinfer1::Weights> nid2weights;
  // extra weights that come from converting a layer of TVM to TensorRT.
  // For example, batch_norm is implemented as Scale in TensorRT.
  // We need to convert batch norm weights to the weights of Scale in TensorRT.
  std::vector<nvinfer1::Weights> extra_weights;
  // maps nid to input data
  std::unordered_map<uint32_t, nvinfer1::ITensor*> nid2tensor;
  // maps nid to layer
  std::unordered_map<uint32_t, nvinfer1::ILayer*> nid2layer;
  nvinfer1::INetworkDefinition* network = infer_engine_builder_->createNetwork();
  CHECK(network != nullptr) << "Creating TensorRT network failed";
  const auto& nodes = subgraph.nodes;
  for (size_t nid = 0; nid < nodes.size(); ++nid) {
    if (nodes[nid].op_name == "null") continue;  // variable node
    auto it = add_trt_layer_funcs.find(nodes[nid].op_name);
    CHECK(it != add_trt_layer_funcs.end()) << "Unsupported operator conversion to TRT, op name: "
        << nodes[nid].op_name;
    it->second(nid, nodes, input_nid2idx, data_entries, network, &nid2weights,
               &nid2tensor, &nid2layer, input_data_idx, input_data_names, &extra_weights);
  }
  // mark output layers
  std::unordered_set<uint32_t> head_entry_ids;  // deduplicate output entries
  for (const auto& output_entry : subgraph.heads) {
    const uint32_t eid = subgraph.entry_id(output_entry);
    auto it = head_entry_ids.find(eid);
    CHECK(it == head_entry_ids.end()) << "Subgraph node cannot have duplicate output entries";
    CHECK(nid2layer.count(output_entry.node_id));
    nvinfer1::ILayer* output_layer = nid2layer.at(output_entry.node_id);
    output_names->push_back(
        nodes[output_entry.node_id].node_name + "_output" + std::to_string(output_entry.index));
    output_layer->getOutput(output_entry.index)->setName(output_names->back().c_str());
    network->markOutput(*(output_layer->getOutput(output_entry.index)));
    head_entry_ids.emplace(eid);
  }

  CHECK(!input_data_idx->empty());
  const int batch_size = data_entries[input_data_idx->at(0)].shape[0];
  // build engine
  infer_engine_builder_->setMaxBatchSize(batch_size);
  infer_engine_builder_->setMaxWorkspaceSize(max_workspace_size_);
  infer_engine_builder_->setFp16Mode(use_fp16_);
  nvinfer1::ICudaEngine* engine = infer_engine_builder_->buildCudaEngine(*network);
  CHECK(engine != nullptr);

  // clean up
  network->destroy();
  for (auto& kv : nid2weights) {
    free(const_cast<void*>(kv.second.values));
  }
  for (auto& w : extra_weights) {
    free(const_cast<void*>(w.values));
  }
  return engine;
}

}  // namespace contrib
}  // namespace tvm
