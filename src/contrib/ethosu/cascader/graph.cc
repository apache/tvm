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
#include "graph.h"

#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common.h"
#include "stripe_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void PerformanceInfoNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("_compute_cycles", &compute_cycles);
  Array<IntImm> tmp_reads = make_array(read_bytes);
  v->Visit("_read_bytes", &tmp_reads);
  v->Visit("_write_bytes", &write_bytes);
  v->Visit("_block_config", &block_config);
}

TVM_REGISTER_NODE_TYPE(PerformanceInfoNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PerformanceInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PerformanceInfoNode*>(ref.get());
      p->stream << "PerformanceInfo(compute_cycles=" << node->compute_cycles << ", read_bytes=[";
      for (auto rb : node->read_bytes) {
        p->stream << rb << ", ";
      }
      p->stream << "], write_bytes=" << node->write_bytes << ")";
    });

void TensorNode::VisitAttrs(AttrVisitor* v) {
  Array<Integer> tmp_arr = make_array(shape_);
  v->Visit("_shape", &tmp_arr);
  v->Visit("_dtype", &dtype_);
  v->Visit("_is_constant", &is_constant_);
  double compression_ratio = static_cast<double>(compression_ratio_);
  v->Visit("_compression_ratio", &compression_ratio);
  Array<Part> tmp_prods(producers_);
  v->Visit("_producers", &tmp_prods);
  Array<Part> tmp_cons(consumers_);
  v->Visit("_consumers", &tmp_cons);
  v->Visit("_size", &size_);
}

Tensor::Tensor(const std::vector<int>& shape, DataType dtype, bool is_constant = false,
               float compression_ratio = 1.0) {
  auto n = make_object<TensorNode>();
  n->shape_ = std::move(shape);
  n->dtype_ = dtype;
  n->is_constant_ = is_constant;
  n->compression_ratio_ = compression_ratio;
  n->size_ = mul_reduce(n->shape_) * n->dtype_.bytes();
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.Tensor")
    .set_body_typed([](Array<Integer> shape, DataType dtype, bool is_constant,
                       double compression_ratio) {
      std::vector<int> vshape = make_vector<int, Integer>(shape);
      return Tensor(vshape, dtype, is_constant, compression_ratio);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.TensorAddProducer")
    .set_body_method<Tensor>(&TensorNode::AddProducer);
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.TensorAddConsumer")
    .set_body_method<Tensor>(&TensorNode::AddConsumer);

TVM_REGISTER_NODE_TYPE(TensorNode);

void PartNode::VisitAttrs(AttrVisitor* v) {
  Array<Propagator> tmp_prp(propagators_);
  v->Visit("_propagators", &tmp_prp);
  Array<Tensor> tmp_ins(input_tensors_);
  v->Visit("_input_tensors", &tmp_ins);
  v->Visit("_output_tensor", &output_tensor_);
  v->Visit("_in_line", &in_line_);
  Array<te::Tensor> tmp_te_ins(subgraph_.input_tensors);
  v->Visit("_te_input_tensors", &tmp_te_ins);
  v->Visit("_te_output_tensor", &subgraph_.output_tensor);
}

void PartNode::SetInput(uint64_t input_index, const Tensor& input_tensor) {
  ICHECK_LT(input_index, input_tensors_.size());
  input_tensors_[input_index] = std::move(input_tensor);
}

std::vector<StripeConfig> PartNode::CalculateInputStripeConfigs(
    const StripeConfig& output_stripe_config) {
  std::vector<StripeConfig> input_stripe_configs;
  for (const auto& propagator : propagators_) {
    input_stripe_configs.push_back(propagator->propagate(output_stripe_config));
  }
  return input_stripe_configs;
}

const std::vector<int> PartNode::GetStripeAlignHint() const {
  ICHECK_GT(propagators_.size(), 0);
  size_t dims = propagators_[0]->GetOutputDims();
  std::vector<int> compute_quantum(dims);
  for (size_t i = 0; i < dims; i++) {
    compute_quantum[i] = 1;
  }
  return compute_quantum;
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PartSetInput")
    .set_body_method<Part>(&PartNode::SetInput);
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PartSetOutput")
    .set_body_method<Part>(&PartNode::SetOutput);
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PartCalculateInputStripeConfigs")
    .set_body_typed([](Part part, StripeConfig output_stripe_config) {
      auto input_stripe_configs = part->CalculateInputStripeConfigs(output_stripe_config);
      return Array<StripeConfig>(input_stripe_configs);
    });
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PartGetStripeAlignHint").set_body_typed([](Part part) {
  std::vector<int> align_hint = part->GetStripeAlignHint();
  return make_array(align_hint);
});
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PartGetPerformanceInfo")
    .set_body_typed([](Part part, StripeConfig stripe_config, int buffer_mode) {
      BufferMode ebuffer_mode = static_cast<BufferMode>(buffer_mode);
      return part->GetPerformanceInfo(stripe_config, ebuffer_mode);
    });

CascaderGraphNode::CascaderGraphNode(std::vector<Tensor> input_tensors,
                                     std::vector<Tensor> output_tensors)
    : input_tensors_(input_tensors), output_tensors_(output_tensors) {
  Init_();
}

bool VisitedInputs(
    const Part& part,
    const std::unordered_set<Tensor, ObjectPtrHash, ObjectPtrEqual>& visited_tensors) {
  for (const auto& input_tensor : part->GetInputTensors()) {
    if (visited_tensors.find(input_tensor) == visited_tensors.end()) {
      return false;
    }
  }
  return true;
}

void CascaderGraphNode::Init_() {
  std::stack<Tensor> stack;
  std::unordered_set<Tensor, ObjectPtrHash, ObjectPtrEqual> visited_tensors;
  std::unordered_set<Part, ObjectPtrHash, ObjectPtrEqual> visited_parts;
  for (const auto& input : input_tensors_) {
    stack.push(input);
  }
  // Visit the Parts/Tensors in depth-first order using a non-recursive algorithm
  while (!stack.empty()) {
    Tensor tensor = stack.top();
    stack.pop();
    if (visited_tensors.find(tensor) == visited_tensors.end()) {
      visited_tensors.insert(tensor);
      tensor_order_.push_back(tensor);
      for (const auto& part : tensor->GetConsumers()) {
        if (visited_parts.find(part) == visited_parts.end()) {
          // Only visit a Part once we've visited all its input Tensors
          if (!VisitedInputs(part, visited_tensors)) continue;
          visited_parts.insert(part);
          part_order_.push_back(part);
          stack.push(part->GetOutputTensor());
        }
      }
    }
  }
  std::reverse(tensor_order_.begin(), tensor_order_.end());
  std::reverse(part_order_.begin(), part_order_.end());
  int id = 0;
  for (const auto& part : part_order_) {
    part_id_map_[part] = id;
    id++;
  }
  id = 0;
  for (const auto& tensor : tensor_order_) {
    tensor_id_map_[tensor] = id;
    id++;
  }
}

void CascaderGraphNode::VisitAttrs(AttrVisitor* v) {
  Array<Tensor> tmp_ins(input_tensors_);
  v->Visit("_input_tensors", &tmp_ins);
  Array<Tensor> tmp_outs(output_tensors_);
  v->Visit("_output_tensors", &tmp_outs);
  Array<Part> tmp_parr(part_order_);
  v->Visit("_part_order", &tmp_parr);
  Array<Tensor> tmp_tarr(tensor_order_);
  v->Visit("_tensor_order", &tmp_tarr);
}

int CascaderGraphNode::GetPartID(const Part& part) const {
  if (part_id_map_.find(part) == part_id_map_.end()) {
    return -1;
  }
  return part_id_map_.at(part);
}

int CascaderGraphNode::GetTensorID(const Tensor& tensor) const {
  if (tensor_id_map_.find(tensor) == tensor_id_map_.end()) {
    return -1;
  }
  return tensor_id_map_.at(tensor);
}

CascaderGraph::CascaderGraph(std::vector<Tensor> input_tensors,
                             std::vector<Tensor> output_tensors) {
  auto n = make_object<CascaderGraphNode>(input_tensors, output_tensors);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.CascaderGraph")
    .set_body_typed([](Array<Tensor> input_tensors, Array<Tensor> output_tensors) {
      std::vector<Tensor> vinput_tensors(input_tensors.begin(), input_tensors.end());
      std::vector<Tensor> voutput_tensors(output_tensors.begin(), output_tensors.end());
      return CascaderGraph(vinput_tensors, voutput_tensors);
    });
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.CascaderGraphGetPartID")
    .set_body_method<CascaderGraph>(&CascaderGraphNode::GetPartID);
TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.CascaderGraphGetTensorID")
    .set_body_method<CascaderGraph>(&CascaderGraphNode::GetTensorID);

TVM_REGISTER_NODE_TYPE(CascaderGraphNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
