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
 * \file src/contrib/msc/core/ir/graph.cc
 */

#include "graph.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>

#include "../printer/prototxt_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

MSCTensor::MSCTensor(const String& name, const DataType& dtype, const String& layout,
                     const Array<Integer>& shape, const String& alias, const Array<String>& prims) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->name = std::move(name);
  n->alias = std::move(alias);
  n->dtype = std::move(dtype);
  n->shape = std::move(shape);
  n->layout = tvm::tir::Layout(layout);
  n->prims = prims;
  data_ = std::move(n);
}

MSCTensor::MSCTensor(const JsonMSCTensor& j_tensor) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->FromJson(j_tensor);
  data_ = std::move(n);
}

MSCTensor::MSCTensor(const std::string& json_str) {
  ObjectPtr<MSCTensorNode> n = make_object<MSCTensorNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonMSCTensor MSCTensorNode::ToJson() const {
  JsonMSCTensor j_tensor;
  j_tensor.name = name;
  j_tensor.alias = alias;
  j_tensor.dtype = runtime::DLDataType2String(dtype);
  if (layout.defined()) {
    j_tensor.layout = layout.name();
  }
  for (const auto& s : shape) {
    j_tensor.shape.push_back(s->value);
  }
  for (const auto& p : prims) {
    j_tensor.prims.push_back(p);
  }
  return j_tensor;
}

void MSCTensorNode::FromJson(const JsonMSCTensor& j_tensor) {
  name = j_tensor.name;
  alias = j_tensor.alias;
  dtype = DataType(runtime::String2DLDataType(j_tensor.dtype));
  if (j_tensor.layout.size() > 0) {
    layout = tvm::tir::Layout(j_tensor.layout);
  }
  for (const auto& s : j_tensor.shape) {
    shape.push_back(s);
  }
  for (const auto& p : j_tensor.prims) {
    prims.push_back(p);
  }
}

void MSCTensorNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCTensor j_tensor;
  reader.Read(&j_tensor);
  FromJson(j_tensor);
}

size_t MSCTensorNode::Ndim() const { return shape.size(); }

const Integer MSCTensorNode::DimAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, Ndim());
  return shape[v_index];
}

const Integer MSCTensorNode::DimAt(const String& axis) const {
  auto index = layout.IndexOf(tvm::tir::LayoutAxis::Get(axis));
  return DimAt(index);
}

const String MSCTensorNode::PrimAt(int index) const {
  if (prims.size() == 0) {
    return "";
  }
  return prims[CommonUtils::GetIndex(index, Ndim())];
}

const String MSCTensorNode::PrimAt(const String& axis) const {
  return PrimAt(layout.IndexOf(tvm::tir::LayoutAxis::Get(axis)));
}

int32_t MSCTensorNode::LayoutOf(const String& axis) const {
  return layout.IndexOf(tvm::tir::LayoutAxis::Get(axis));
}

const Integer MSCTensorNode::GetSize() const {
  Integer size = Integer(1);
  for (const auto& s : shape) {
    size *= s;
  }
  return size;
}

const String MSCTensorNode::DTypeName() const { return runtime::DLDataType2String(dtype); }

size_t BaseJointNode::AddChild(const BaseJoint& child) const {
  for (size_t i = 0; i < children.size(); i++) {
    if (Downcast<BaseJoint>(children[i])->name == child->name) {
      return i;
    }
  }
  children.push_back(child);
  return children.size() - 1;
}

const BaseJoint BaseJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<BaseJoint>(parents[v_index]);
}

const BaseJoint BaseJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<BaseJoint>(children[v_index]);
}

bool BaseJointNode::HasAttr(const String& key) const { return attrs.count(key); }

bool BaseJointNode::GetAttr(const String& key, std::string* val) const {
  if (attrs.count(key) && attrs[key].size() > 0) {
    *val = attrs[key];
    return true;
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, int* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos > 0) {
      return false;
    }
    try {
      *val = std::stoi(val_str);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, int64_t* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      *val = std::stoi(val_str);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, float* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      *val = std::atof(val_str.c_str());
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, bool* val) const {
  int val_int;
  if (GetAttr(key, &val_int)) {
    *val = (val_int != 0);
    return true;
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<std::string>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::string(s));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<int>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::stoi(std::string(s)));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<int64_t>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    try {
      int pos = val_str.find(",");
      if (pos < 0) {
        return false;
      }
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::stol(std::string(s)));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}
bool BaseJointNode::GetAttr(const String& key, std::vector<float>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::atof(std::string(s).c_str()));
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

bool BaseJointNode::GetAttr(const String& key, std::vector<bool>* val) const {
  std::string val_str;
  if (GetAttr(key, &val_str)) {
    int pos = val_str.find(",");
    if (pos < 0) {
      return false;
    }
    try {
      for (const auto& s : StringUtils::Split(val_str, ",")) {
        (*val).push_back(std::stoi(s) != 0);
      }
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
  return false;
}

MSCJoint::MSCJoint(int index, const String& name, const String& shared_ref, const String& optype,
                   const Map<String, String>& attrs, const Array<String>& scope,
                   const std::vector<std::pair<BaseJoint, size_t>>& inputs,
                   const Array<MSCTensor>& outputs, const Map<String, MSCTensor>& weights) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->index = index;
  n->name = std::move(name);
  n->shared_ref = std::move(shared_ref);
  n->optype = std::move(optype);
  n->attrs = std::move(attrs);
  n->scope = std::move(scope);
  Array<ObjectRef> parents;
  Array<Array<Integer>> array_inputs;
  Array<String> added_parents;
  for (const auto& pair : inputs) {
    // const auto& parent=Downcast<BaseJoint>(pair.first);
    const auto& p_name = pair.first->name;
    int p_idx = -1;
    for (size_t i = 0; i < added_parents.size(); i++) {
      if (added_parents[i] == p_name) {
        p_idx = i;
        break;
      }
    }
    if (p_idx == -1) {
      parents.push_back(pair.first);
      added_parents.push_back(p_name);
      p_idx = added_parents.size() - 1;
    }
    Array<Integer> input{Integer(p_idx), Integer(pair.second)};
    array_inputs.push_back(input);
  }
  n->parents = std::move(parents);
  n->inputs = std::move(array_inputs);
  n->outputs = std::move(outputs);
  n->weights = std::move(weights);
  data_ = std::move(n);
}

MSCJoint::MSCJoint(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->FromJson(j_joint, nodes);
  data_ = std::move(n);
}

MSCJoint::MSCJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<MSCJointNode> n = make_object<MSCJointNode>();
  n->FromJson(json_str, nodes);
  data_ = std::move(n);
}

const MSCJoint MSCJoint::Clone(const MSCJoint& node,
                               const std::vector<std::pair<BaseJoint, size_t>>& inputs) {
  return MSCJoint(node->index, node->name, node->shared_ref, node->optype, node->attrs, node->scope,
                  inputs, node->outputs, node->weights);
}

const JsonMSCJoint MSCJointNode::ToJson() const {
  JsonMSCJoint j_joint;
  j_joint.index = index;
  j_joint.name = name;
  j_joint.shared_ref = shared_ref;
  j_joint.optype = optype;
  for (const auto& pair : attrs) {
    j_joint.attrs[pair.first] = pair.second;
  }
  for (const auto& s : scope) {
    j_joint.scope.push_back(s);
  }
  for (const auto& p : parents) {
    j_joint.parents.push_back(Downcast<BaseJoint>(p)->name);
  }
  for (const auto& i : GetInputs()) {
    j_joint.inputs.push_back(i->name);
  }
  for (const auto& o : GetOutputs()) {
    j_joint.outputs.push_back(o->ToJson());
  }
  for (const auto& pair : weights) {
    j_joint.weights[pair.first] = pair.second->ToJson();
  }
  return j_joint;
}

void MSCJointNode::FromJson(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes) {
  index = j_joint.index;
  name = j_joint.name;
  shared_ref = j_joint.shared_ref;
  optype = j_joint.optype;
  for (const auto& pair : j_joint.attrs) {
    attrs.Set(pair.first, pair.second);
  }
  for (const auto& s : j_joint.scope) {
    scope.push_back(s);
  }
  for (const auto& p_name : j_joint.parents) {
    ICHECK(nodes.count(p_name)) << "Can not find parent " << p_name;
    parents.push_back(nodes[p_name]);
  }
  for (const auto& in_name : j_joint.inputs) {
    String producer, index_str;
    std::tie(producer, index_str) = StringUtils::SplitOnce(in_name, ":");
    int p_idx = -1;
    for (size_t i = 0; i < parents.size(); i++) {
      if (ParentAt(i)->name == producer) {
        p_idx = i;
        break;
      }
    }
    ICHECK(p_idx >= 0) << "Can not find parent for " << in_name;
    Array<Integer> input{Integer(p_idx), Integer(std::stol(index_str))};
    inputs.push_back(input);
  }
  for (const auto& o : j_joint.outputs) {
    outputs.push_back(MSCTensor(o));
  }
  for (const auto& pair : j_joint.weights) {
    weights.Set(pair.first, MSCTensor(pair.second));
  }
}

void MSCJointNode::FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCJoint j_joint;
  reader.Read(&j_joint);
  FromJson(j_joint, nodes);
}

const MSCTensor MSCJointNode::InputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  const auto& out_idx = inputs[v_index][1];
  return ParentAt(p_idx->value)->OutputAt(out_idx->value);
}

const Array<MSCTensor> MSCJointNode::GetInputs() const {
  Array<MSCTensor> t_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    t_inputs.push_back(InputAt(i));
  }
  return t_inputs;
}

const MSCTensor MSCJointNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, outputs.size());
  return outputs[v_index];
}

const Array<MSCTensor> MSCJointNode::GetOutputs() const {
  Array<MSCTensor> t_outputs;
  for (size_t i = 0; i < outputs.size(); i++) {
    t_outputs.push_back(OutputAt(i));
  }
  return t_outputs;
}

const MSCTensor MSCJointNode::WeightAt(const String& wtype) const {
  ICHECK(weights.count(wtype)) << "Can not find " << wtype << " from weights";
  return weights[wtype];
}

const MSCJoint MSCJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<MSCJoint>(parents[v_index]);
}

const MSCJoint MSCJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<MSCJoint>(children[v_index]);
}

const MSCJoint MSCJointNode::ProducerOf(int index) const {
  const auto& pair = ProducerAndIdxOf(index);
  return pair.first;
}

const MSCJoint MSCJointNode::ProducerOf(const String& input_name) const {
  const auto& pair = ProducerAndIdxOf(input_name);
  return pair.first;
}

const MSCJoint MSCJointNode::ProducerOf(const MSCTensor& input) const {
  return ProducerOf(input->name);
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, inputs.size());
  const auto& p_idx = inputs[v_index][0];
  return std::make_pair(ParentAt(p_idx->value), inputs[v_index][1]->value);
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(const String& name) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (InputAt(i)->name == name) {
      return ProducerAndIdxOf(i);
    }
  }
  LOG(FATAL) << "Can not find producer of " << name;
}

const std::pair<MSCJoint, size_t> MSCJointNode::ProducerAndIdxOf(const MSCTensor& input) const {
  return ProducerAndIdxOf(input->name);
}

MSCPrim::MSCPrim(int index, const String& name, const String& optype,
                 const Array<BaseJoint>& parents, const Map<String, String>& attrs) {
  ObjectPtr<MSCPrimNode> n = make_object<MSCPrimNode>();
  n->index = index;
  n->name = std::move(name);
  n->optype = std::move(optype);
  n->attrs = std::move(attrs);
  for (const auto& p : parents) {
    n->parents.push_back(p);
  }
  data_ = std::move(n);
}

MSCPrim::MSCPrim(const JsonMSCPrim& j_prim, const Map<String, BaseJoint>& prims) {
  ObjectPtr<MSCPrimNode> n = make_object<MSCPrimNode>();
  n->FromJson(j_prim, prims);
  data_ = std::move(n);
}

MSCPrim::MSCPrim(const std::string& json_str, const Map<String, BaseJoint>& prims) {
  ObjectPtr<MSCPrimNode> n = make_object<MSCPrimNode>();
  n->FromJson(json_str, prims);
  data_ = std::move(n);
}

const JsonMSCPrim MSCPrimNode::ToJson() const {
  JsonMSCPrim j_prim;
  j_prim.index = index;
  j_prim.name = name;
  j_prim.optype = optype;
  for (const auto& pair : attrs) {
    j_prim.attrs[pair.first] = pair.second;
  }
  for (const auto& p : parents) {
    j_prim.parents.push_back(Downcast<BaseJoint>(p)->name);
  }
  return j_prim;
}

void MSCPrimNode::FromJson(const JsonMSCPrim& j_prim, const Map<String, BaseJoint>& prims) {
  index = j_prim.index;
  name = j_prim.name;
  optype = j_prim.optype;
  for (const auto& pair : j_prim.attrs) {
    attrs.Set(pair.first, pair.second);
  }
  for (const auto& p_name : j_prim.parents) {
    ICHECK(prims.count(p_name)) << "Can not find parent " << p_name;
    parents.push_back(prims[p_name]);
  }
}

void MSCPrimNode::FromJson(const std::string& json_str, const Map<String, BaseJoint>& prims) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCPrim j_prim;
  reader.Read(&j_prim);
  FromJson(j_prim, prims);
}

const MSCPrim MSCPrimNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<MSCPrim>(parents[v_index]);
}

const MSCPrim MSCPrimNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<MSCPrim>(children[v_index]);
}

WeightJoint::WeightJoint(int index, const String& name, const String& shared_ref,
                         const String& weight_type, const MSCTensor& weight,
                         const Array<BaseJoint> parents, const Map<String, String>& attrs,
                         const Array<BaseJoint>& friends) {
  ObjectPtr<WeightJointNode> n = make_object<WeightJointNode>();
  n->index = index;
  n->name = std::move(name);
  n->shared_ref = std::move(shared_ref);
  n->weight_type = std::move(weight_type);
  n->attrs = std::move(attrs);
  n->weight = std::move(weight);
  for (const auto& p : parents) {
    n->parents.push_back(p);
  }
  n->friends = std::move(friends);
  data_ = std::move(n);
}

WeightJoint::WeightJoint(const JsonWeightJoint& j_joint, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<WeightJointNode> n = make_object<WeightJointNode>();
  n->FromJson(j_joint, nodes);
  data_ = std::move(n);
}

WeightJoint::WeightJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  ObjectPtr<WeightJointNode> n = make_object<WeightJointNode>();
  n->FromJson(json_str, nodes);
  data_ = std::move(n);
}

const JsonWeightJoint WeightJointNode::ToJson() const {
  JsonWeightJoint j_joint;
  j_joint.index = index;
  j_joint.name = name;
  j_joint.shared_ref = shared_ref;
  j_joint.weight_type = weight_type;
  j_joint.weight = weight->ToJson();
  for (const auto& pair : attrs) {
    j_joint.attrs[pair.first] = pair.second;
  }
  for (const auto& p : parents) {
    j_joint.parents.push_back(Downcast<BaseJoint>(p)->name);
  }
  for (const auto& f : friends) {
    j_joint.friends.push_back(Downcast<BaseJoint>(f)->name);
  }

  return j_joint;
}

void WeightJointNode::FromJson(const JsonWeightJoint& j_joint,
                               const Map<String, BaseJoint>& nodes) {
  index = j_joint.index;
  name = j_joint.name;
  shared_ref = j_joint.shared_ref;
  weight_type = j_joint.weight_type;
  weight = MSCTensor(j_joint.weight);
  for (const auto& pair : j_joint.attrs) {
    attrs.Set(pair.first, pair.second);
  }
  for (const auto& p_name : j_joint.parents) {
    ICHECK(nodes.count(p_name)) << "Can not find parent " << p_name;
    parents.push_back(nodes[p_name]);
  }
}

void WeightJointNode::FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonWeightJoint j_joint;
  reader.Read(&j_joint);
  FromJson(j_joint, nodes);
}

const WeightJoint WeightJointNode::ParentAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, parents.size());
  return Downcast<WeightJoint>(parents[v_index]);
}

const WeightJoint WeightJointNode::ChildAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, children.size());
  return Downcast<WeightJoint>(children[v_index]);
}

const bool BaseGraphNode::HasNode(const String& name) const {
  return nodes.count(name) ? true : false;
}

MSCGraph::MSCGraph(const String& name, const Array<MSCJoint>& nodes,
                   const Array<String>& input_names, const Array<String>& output_names,
                   const Array<MSCPrim>& prims) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->name = std::move(name);
  for (const auto& node : nodes) {
    n->node_names.push_back(node->name);
    n->nodes.Set(node->name, node);
  }
  n->input_names = std::move(input_names);
  n->output_names = std::move(output_names);
  for (const auto& prim : prims) {
    n->prim_names.push_back(prim->name);
    n->prims.Set(prim->name, prim);
  }
  n->AnalysisGraph();
  data_ = std::move(n);
}

MSCGraph::MSCGraph(const JsonMSCGraph& j_graph) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->FromJson(j_graph);
  data_ = std::move(n);
}

MSCGraph::MSCGraph(const std::string& json_str) {
  ObjectPtr<MSCGraphNode> n = make_object<MSCGraphNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonMSCGraph MSCGraphNode::ToJson() const {
  JsonMSCGraph j_graph;
  j_graph.name = name;
  for (const auto& i : input_names) {
    j_graph.inputs.push_back(i);
  }
  for (const auto& o : output_names) {
    j_graph.outputs.push_back(o);
  }
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    j_graph.nodes.push_back(node->ToJson());
  }
  for (const auto& n : prim_names) {
    const auto& prim = FindPrim(n);
    j_graph.prims.push_back(prim->ToJson());
  }
  return j_graph;
}

void MSCGraphNode::FromJson(const JsonMSCGraph& j_graph) {
  name = j_graph.name;
  for (const auto& i : j_graph.inputs) {
    input_names.push_back(i);
  }
  for (const auto& o : j_graph.outputs) {
    output_names.push_back(o);
  }
  Map<String, BaseJoint> loaded_nodes;
  for (const auto& n : j_graph.nodes) {
    const auto& node = MSCJoint(n, loaded_nodes);
    loaded_nodes.Set(node->name, node);
    for (const auto& p : node->parents) {
      Downcast<BaseJoint>(p)->AddChild(node);
    }
    node_names.push_back(node->name);
    nodes.Set(node->name, node);
  }
  Map<String, BaseJoint> loaded_prims;
  for (const auto& n : j_graph.prims) {
    const auto& prim = MSCPrim(n, loaded_prims);
    loaded_prims.Set(prim->name, prim);
    for (const auto& p : prim->parents) {
      Downcast<BaseJoint>(p)->AddChild(prim);
    }
    prim_names.push_back(prim->name);
    prims.Set(prim->name, prim);
  }
  AnalysisGraph();
}

void MSCGraphNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonMSCGraph j_graph;
  reader.Read(&j_graph);
  FromJson(j_graph);
}

const String MSCGraphNode::ToPrototxt() const {
  PrototxtPrinter printer;
  printer.Append(Map<String, ObjectRef>{{"name", name}});
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    // define layer
    std::vector<std::pair<String, ObjectRef>> layer;
    layer.push_back(std::make_pair("name", node->name));
    layer.push_back(std::make_pair("type", StringUtils::Replace(node->optype, ".", "_")));
    layer.push_back(std::make_pair("top", node->name));
    for (const auto& p : node->parents) {
      layer.push_back(std::make_pair("bottom", Downcast<BaseJoint>(p)->name));
    }
    // define layer param
    Map<String, ObjectRef> param;
    param.Set("idx", Integer(node->index));
    for (size_t i = 0; i < node->inputs.size(); i++) {
      param.Set("input_" + std::to_string(i), node->InputAt(i));
    }
    for (size_t i = 0; i < node->outputs.size(); i++) {
      param.Set("output_" + std::to_string(i), node->OutputAt(i));
    }
    for (const auto& pair : node->weights) {
      param.Set("param_" + pair.first, pair.second);
    }
    for (const auto& pair : node->attrs) {
      param.Set(pair.first, pair.second);
    }
    layer.push_back(std::make_pair("layer_param", PrototxtPrinter::ToDictDoc(param)));
    // Append the layer Map
    printer.Append(Map<String, ObjectRef>{{"layer", PrototxtPrinter::ToDictDoc(layer)}});
  }
  return printer.GetString();
}

const MSCJoint MSCGraphNode::FindNode(const String& name) const {
  ICHECK(nodes.count(name)) << "Can not find node " << name;
  return Downcast<MSCJoint>(nodes[name]);
}

const MSCPrim MSCGraphNode::FindPrim(const String& name) const {
  ICHECK(prims.count(name)) << "Can not find prim " << name;
  return prims[name];
}

const MSCTensor MSCGraphNode::InputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, input_names.size());
  return FindTensor(input_names[v_index]);
}

const Array<MSCTensor> MSCGraphNode::GetInputs() const {
  Array<MSCTensor> t_inputs;
  for (size_t i = 0; i < input_names.size(); i++) {
    t_inputs.push_back(InputAt(i));
  }
  return t_inputs;
}

const MSCTensor MSCGraphNode::OutputAt(int index) const {
  size_t v_index = CommonUtils::GetIndex(index, output_names.size());
  return FindTensor(output_names[v_index]);
}

const Array<MSCTensor> MSCGraphNode::GetOutputs() const {
  Array<MSCTensor> t_outputs;
  for (size_t i = 0; i < output_names.size(); i++) {
    t_outputs.push_back(OutputAt(i));
  }
  return t_outputs;
}

const Array<MSCJoint> MSCGraphNode::GetEntries() const {
  Array<MSCJoint> entries;
  for (size_t i = 0; i < input_names.size(); i++) {
    entries.push_back(FindProducer(input_names[i]));
  }
  return entries;
}

const Array<MSCJoint> MSCGraphNode::GetExits() const {
  Array<MSCJoint> exits;
  std::set<String> setted_exits;
  for (size_t i = 0; i < output_names.size(); i++) {
    const auto& exit = FindProducer(output_names[i]);
    if (setted_exits.count(exit->name)) {
      continue;
    }
    exits.push_back(exit);
    setted_exits.insert(exit->name);
  }
  return exits;
}

const bool MSCGraphNode::HasTensor(const String& name) const {
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  if (weight_holders.count(tensor_name)) {
    return true;
  }
  String host, index;
  std::tie(host, index) = StringUtils::SplitOnce(tensor_name, ":");
  return nodes.count(host) > 0 ? true : false;
}

const MSCTensor MSCGraphNode::FindTensor(const String& name) const {
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  if (weight_holders.count(tensor_name)) {
    const auto& node = FindNode(weight_holders[tensor_name][0]);
    for (const auto& pair : node->weights) {
      if (pair.second->name == tensor_name) {
        return pair.second;
      }
    }
    LOG(FATAL) << "Can not find weight " << name << " from " << node;
  }
  const auto& pair = FindProducerAndIdx(name);
  return pair.first->OutputAt(pair.second);
}

const MSCJoint MSCGraphNode::FindProducer(const String& name) const {
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  if (weight_holders.count(tensor_name)) {
    return FindNode(weight_holders[tensor_name][0]);
  }
  const auto& pair = FindProducerAndIdx(name);
  return pair.first;
}

const MSCJoint MSCGraphNode::FindProducer(const MSCTensor& tensor) const {
  return FindProducer(tensor->name);
}

const std::pair<MSCJoint, size_t> MSCGraphNode::FindProducerAndIdx(const String& name) const {
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  ICHECK(!weight_holders.count(tensor_name)) << "Weight " << name << " has no producer with index";
  String host, index;
  std::tie(host, index) = StringUtils::SplitOnce(tensor_name, ":");
  if (index.size() == 0) {
    const auto& node = FindNode(host);
    ICHECK(node->optype == "constant") << "Tensor without index should be constant, get " << node;
    return std::make_pair(node, 0);
  }
  return std::make_pair(FindNode(host), std::stoi(index));
}

const std::pair<MSCJoint, size_t> MSCGraphNode::FindProducerAndIdx(const MSCTensor& tensor) const {
  return FindProducerAndIdx(tensor->name);
}

const Array<MSCJoint> MSCGraphNode::FindConsumers(const String& name) const {
  Array<MSCJoint> consumers;
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  if (weight_holders.count(tensor_name)) {
    for (const auto& h : weight_holders[tensor_name]) {
      consumers.push_back(FindNode(h));
    }
  } else {
    const auto& producer = FindProducer(name);
    for (const auto& c : producer->children) {
      consumers.push_back(Downcast<MSCJoint>(c));
    }
  }
  return consumers;
}

const Array<MSCJoint> MSCGraphNode::FindConsumers(const MSCTensor& tensor) const {
  return FindConsumers(tensor->name);
}

const std::vector<std::pair<MSCJoint, size_t>> MSCGraphNode::FindConsumersAndIndices(
    const String& name) const {
  const String& tensor_name = tensor_alias.count(name) ? tensor_alias[name] : name;
  ICHECK(!weight_holders.count(tensor_name)) << "Weight has no index";
  std::vector<std::pair<MSCJoint, size_t>> consumers;
  for (const auto& c : FindConsumers(name)) {
    bool find_tensor = false;
    for (size_t i = 0; i < c->inputs.size(); i++) {
      if (c->InputAt(i)->name == name) {
        consumers.push_back(std::make_pair(c, i));
        find_tensor = true;
        break;
      }
    }
    ICHECK(find_tensor) << "Can not find tensor " << name << " from " << c;
  }
  return consumers;
}

const std::vector<std::pair<MSCJoint, size_t>> MSCGraphNode::FindConsumersAndIndices(
    const MSCTensor& tensor) const {
  return FindConsumersAndIndices(tensor->name);
}

void MSCGraphNode::AnalysisGraph() {
  // Add children
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    for (const auto& p : node->parents) {
      Downcast<MSCJoint>(p)->AddChild(node);
    }
  }
  // Check inputs and outputs
  for (const auto& i : input_names) {
    const auto& input = FindTensor(i);
    if (input->alias.size() > 0) {
      tensor_alias.Set(input->alias, input->name);
    }
  }
  for (const auto& o : output_names) {
    FindTensor(o);
  }
  // Set tensor alias and weight_holders
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    for (const auto& o : node->outputs) {
      if (o->alias.size() > 0) {
        tensor_alias.Set(o->alias, o->name);
      }
    }
    for (const auto& pair : node->weights) {
      const auto& w_name = pair.second->name;
      if (weight_holders.count(w_name)) {
        Array<String> holders = weight_holders[w_name];
        holders.push_back(n);
        weight_holders.Set(w_name, holders);
      } else {
        weight_holders.Set(w_name, Array<String>({n}));
        if (pair.second->alias.size() > 0) {
          tensor_alias.Set(pair.second->alias, pair.second->name);
        }
      }
    }
  }
}

WeightGraph::WeightGraph(const MSCGraph& graph, const Map<String, Array<String>>& main_wtypes,
                         const Map<String, String>& relation_wtypes) {
  ObjectPtr<WeightGraphNode> n = make_object<WeightGraphNode>();
  n->name = graph->name + "_weights";
  n->Build(graph, main_wtypes, relation_wtypes);
  data_ = std::move(n);
}

WeightGraph::WeightGraph(const JsonWeightGraph& j_graph) {
  ObjectPtr<WeightGraphNode> n = make_object<WeightGraphNode>();
  n->FromJson(j_graph);
  data_ = std::move(n);
}

WeightGraph::WeightGraph(const std::string& json_str) {
  ObjectPtr<WeightGraphNode> n = make_object<WeightGraphNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

void WeightGraphNode::Build(const MSCGraph& graph, const Map<String, Array<String>>& main_wtypes,
                            const Map<String, String>& relation_wtypes) {
  auto sort_nodes = [&graph](const BaseJoint& node_a, const BaseJoint& node_b) {
    return graph->FindProducer(node_a->name)->index < graph->FindProducer(node_b->name)->index;
  };

  auto find_parents = [this, &main_wtypes, &relation_wtypes, &sort_nodes](const MSCJoint& node) {
    std::vector<BaseJoint> parents;
    std::queue<MSCJoint> frontier;
    std::set<MSCJoint> explored;
    for (const auto& p : node->parents) {
      frontier.push(Downcast<MSCJoint>(p));
    }
    while (!frontier.empty()) {
      const auto& current = frontier.front();
      if (explored.count(current)) {
        frontier.pop();
        continue;
      }
      explored.insert(current);
      if (main_wtypes.count(current->optype)) {
        for (const auto& t_type : main_wtypes[current->optype]) {
          if (current->weights.count(t_type)) {
            parents.push_back(FindNode(current->WeightAt(t_type)->name));
          }
        }
      } else if (relation_wtypes.count(current->optype)) {
        parents.push_back(FindNode(current->OutputAt(0)->name));
      } else {
        for (const auto& p : current->parents) {
          const auto& new_parent = Downcast<MSCJoint>(p);
          if (!explored.count(new_parent)) {
            frontier.push(new_parent);
          }
        }
      }
      frontier.pop();
    }
    Array<BaseJoint> parents_array;
    if (parents.size() > 1) {
      std::sort(parents.begin(), parents.end(), sort_nodes);
    }
    for (const auto& p : parents) {
      parents_array.push_back(p);
    }
    return parents_array;
  };

  for (const auto& n : graph->node_names) {
    const auto& node = graph->FindNode(n);
    if (node->shared_ref.size() > 0) {
      continue;
    }
    if (main_wtypes.count(node->optype) || relation_wtypes.count(node->optype) ||
        node->weights.size() > 0) {
      const auto& w_parents = find_parents(node);
      bool bind_friends = true;
      if (relation_wtypes.count(node->optype) && relation_wtypes[node->optype] == "multi_inputs") {
        bind_friends = false;
      }
      if (w_parents.size() > 1 && bind_friends) {
        for (const auto& p : w_parents) {
          Downcast<WeightJoint>(p)->friends = w_parents;
        }
      }
      if (main_wtypes.count(node->optype)) {
        for (const auto& wtype : main_wtypes[node->optype]) {
          if (node->weights.count(wtype)) {
            const auto& weight = node->WeightAt(wtype);
            Map<String, String> attrs;
            attrs.Set("producer_type", node->optype);
            attrs.Set("weight_strategy", "main");
            const auto& w_node =
                WeightJoint(node_names.size(), weight->name, "", wtype, weight, w_parents, attrs);
            for (const auto& p : w_parents) {
              p->AddChild(w_node);
            }
            nodes.Set(weight->name, w_node);
            node_names.push_back(weight->name);
          }
        }
        const BaseJoint& head = FindNode(node_names[node_names.size() - 1]);
        for (const auto& pair : node->weights) {
          if (!nodes.count(pair.second->name)) {
            Map<String, String> attrs;
            attrs.Set("producer_type", node->optype);
            attrs.Set("weight_strategy", "follow");
            const auto& w_node = WeightJoint(node_names.size(), pair.second->name, "", pair.first,
                                             pair.second, {head}, attrs);
            head->AddChild(w_node);
            nodes.Set(pair.second->name, w_node);
            node_names.push_back(pair.second->name);
          }
        }
      } else if (relation_wtypes.count(node->optype)) {
        const auto& tensor = node->OutputAt(0);
        Map<String, String> attrs;
        attrs.Set("producer_type", node->optype);
        if (node->optype == "reshape") {
          // TODO(archermmt): check non-passby reshape
          attrs.Set("weight_strategy", "passby");
        } else {
          attrs.Set("weight_strategy", relation_wtypes[node->optype]);
        }
        const auto& t_node =
            WeightJoint(node_names.size(), tensor->name, "", "output", tensor, w_parents, attrs);
        for (const auto& p : w_parents) {
          p->AddChild(t_node);
        }
        nodes.Set(tensor->name, t_node);
        node_names.push_back(tensor->name);
      } else if (node->weights.size() > 0) {
        for (const auto& pair : node->weights) {
          if (!nodes.count(pair.second->name)) {
            Map<String, String> attrs;
            attrs.Set("producer_type", node->optype);
            attrs.Set("weight_strategy", "follow");
            const auto& w_node = WeightJoint(node_names.size(), pair.second->name, "", pair.first,
                                             pair.second, w_parents, attrs);
            for (const auto& p : w_parents) {
              p->AddChild(w_node);
            }
            nodes.Set(pair.second->name, w_node);
            node_names.push_back(pair.second->name);
          }
        }
      }
    }
  }
}

const WeightJoint WeightGraphNode::FindNode(const String& name) const {
  ICHECK(nodes.count(name)) << "Can not find node " << name;
  return Downcast<WeightJoint>(nodes[name]);
}

const JsonWeightGraph WeightGraphNode::ToJson() const {
  JsonWeightGraph j_graph;
  j_graph.name = name;
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    j_graph.nodes.push_back(node->ToJson());
  }
  return j_graph;
}

void WeightGraphNode::FromJson(const JsonWeightGraph& j_graph) {
  name = j_graph.name;
  Map<String, BaseJoint> loaded_nodes;
  for (const auto& n : j_graph.nodes) {
    const auto& node = WeightJoint(n, loaded_nodes);
    loaded_nodes.Set(node->name, node);
    for (const auto& p : node->parents) {
      Downcast<BaseJoint>(p)->AddChild(node);
    }
    node_names.push_back(node->name);
    nodes.Set(node->name, node);
  }
  // set friends
  for (const auto& j_joint : j_graph.nodes) {
    const auto& node = Downcast<WeightJoint>(nodes[j_joint.name]);
    for (const auto& f_name : j_joint.friends) {
      ICHECK(nodes.count(f_name)) << "Can not find friend " << f_name;
      node->friends.push_back(nodes[f_name]);
    }
  }
}

void WeightGraphNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonWeightGraph j_graph;
  reader.Read(&j_graph);
  FromJson(j_graph);
}

const String WeightGraphNode::ToPrototxt() const {
  PrototxtPrinter printer;
  printer.Append(Map<String, ObjectRef>{{"name", name}});
  for (const auto& n : node_names) {
    const auto& node = FindNode(n);
    // define layer
    std::vector<std::pair<String, ObjectRef>> layer;
    layer.push_back(std::make_pair("name", node->name));
    layer.push_back(std::make_pair("type", node->weight_type));
    layer.push_back(std::make_pair("top", node->name));
    for (const auto& p : node->parents) {
      layer.push_back(std::make_pair("bottom", Downcast<BaseJoint>(p)->name));
    }
    // define layer param
    Map<String, ObjectRef> param;
    param.Set("idx", Integer(node->index));
    param.Set("weight", node->weight);
    for (size_t i = 0; i < node->friends.size(); i++) {
      param.Set("friend_" + std::to_string(i), Downcast<WeightJoint>(node->friends[i]));
    }
    for (const auto& pair : node->attrs) {
      param.Set(pair.first, pair.second);
    }
    layer.push_back(std::make_pair("layer_param", PrototxtPrinter::ToDictDoc(param)));
    // Append the layer Map
    printer.Append(Map<String, ObjectRef>{{"layer", PrototxtPrinter::ToDictDoc(layer)}});
  }
  return printer.GetString();
}

MSCGraph PruneWeights(const MSCGraph& graph, const Map<String, MSCTensor>& pruned_tensors) {
  Array<MSCJoint> nodes;
  std::unordered_map<String, std::pair<BaseJoint, size_t>> inputs_map;
  for (const auto& name : graph->node_names) {
    const auto& node = graph->FindNode(name);
    // define inputs
    std::vector<std::pair<BaseJoint, size_t>> inputs;
    for (const auto& input : node->GetInputs()) {
      ICHECK(inputs_map.count(input->name)) << "Can not find input " << input;
      inputs.push_back(inputs_map[input->name]);
    }
    // define outputs
    Array<MSCTensor> outputs;
    for (const auto& out : node->outputs) {
      const auto& output = pruned_tensors.count(out->name) ? pruned_tensors[out->name] : out;
      outputs.push_back(output);
    }
    // define weights
    Map<String, MSCTensor> weights;
    for (const auto& pair : node->weights) {
      const auto& weight =
          pruned_tensors.count(pair.second->name) ? pruned_tensors[pair.second->name] : pair.second;
      weights.Set(pair.first, weight);
    }
    // define attributes
    Map<String, String> attrs = node->attrs;
    if (node->optype == "reshape" && attrs.count("shape") &&
        pruned_tensors.count(node->OutputAt(0)->name)) {
      const auto& new_shape = pruned_tensors[node->OutputAt(0)->name]->shape;
      attrs.Set("shape", StringUtils::ToString(new_shape));
    }
    // create new node
    const auto& new_node = MSCJoint(static_cast<int>(nodes.size()), node->name, node->shared_ref,
                                    node->optype, attrs, node->scope, inputs, outputs, weights);
    nodes.push_back(new_node);
    for (size_t i = 0; i < new_node->outputs.size(); i++) {
      inputs_map[new_node->OutputAt(i)->name] = std::make_pair(new_node, i);
    }
    for (const auto& p : new_node->parents) {
      Downcast<BaseJoint>(p)->AddChild(new_node);
    }
  }
  Array<MSCPrim> prims;
  for (const auto& name : graph->prim_names) {
    prims.push_back(graph->FindPrim(name));
  }
  return MSCGraph(graph->name, nodes, graph->input_names, graph->output_names, prims);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCTensorNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* tensor = static_cast<const MSCTensorNode*>(node.get());
      p->PrintIndent();
      p->stream << tensor->name;
      if (tensor->alias.size() > 0) {
        p->stream << "(" << tensor->alias << ")";
      }
      p->stream << "<";
      for (size_t i = 0; i < tensor->Ndim(); i++) {
        const auto& prim = tensor->PrimAt(i);
        p->stream << (prim.size() > 0 ? prim : StringUtils::ToString(tensor->shape[i]))
                  << (i == tensor->Ndim() - 1 ? "|" : ",");
      }
      p->stream << tensor->dtype;
      if (tensor->layout.defined()) {
        p->stream << "|" << tensor->layout.name();
      }
      p->stream << ">";
    });

#define MSC_NODE_BASE_HEAD(Stream, Joint, Type)                                          \
  Stream << Type << "_" << Joint->index << " " << Joint->name;                           \
  if (Joint->shared_ref.size() > 0) {                                                    \
    Stream << "(M: " << Joint->shared_ref << ")";                                        \
  }                                                                                      \
  Stream << " <PARENTS: ";                                                               \
  if (Joint->parents.size() > 0) {                                                       \
    for (size_t i = 0; i < Joint->parents.size(); i++) {                                 \
      Stream << Joint->ParentAt(i)->name << (i == Joint->parents.size() - 1 ? "" : ","); \
    }                                                                                    \
  }                                                                                      \
  Stream << "| CHILDERN: ";                                                              \
  if (Joint->children.size() > 0) {                                                      \
    for (size_t i = 0; i < Joint->children.size(); i++) {                                \
      Stream << Joint->ChildAt(i)->name << (i == Joint->children.size() - 1 ? "" : ","); \
    }                                                                                    \
  }                                                                                      \
  Stream << ">\n";

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCJointNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* joint = static_cast<const MSCJointNode*>(node.get());
      p->PrintIndent();
      MSC_NODE_BASE_HEAD(p->stream, joint, "N");
      if (joint->inputs.size() > 0) {
        p->stream << "  IN: ";
        for (size_t i = 0; i < joint->inputs.size(); i++) {
          p->stream << joint->InputAt(i) << (i == joint->inputs.size() - 1 ? "\n" : ",");
        }
      }
      p->stream << "  OUT: ";
      for (size_t i = 0; i < joint->outputs.size(); i++) {
        p->stream << joint->OutputAt(i) << (i == joint->outputs.size() - 1 ? "\n" : ",");
      }
      p->stream << "  OPTYPE: " << joint->optype << "\n";
      if (joint->scope.size() > 0) {
        p->stream << "  SCOPE: ";
        for (size_t i = 0; i < joint->scope.size(); i++) {
          p->stream << joint->scope[i] << (i == joint->scope.size() - 1 ? "\n" : ".");
        }
      }
      if (joint->attrs.size() > 0) {
        p->stream << "  ATTRS: ";
        for (const auto& pair : joint->attrs) {
          p->stream << pair.first << "=" << pair.second << " ";
        }
        p->stream << "\n";
      }
      if (joint->weights.size() > 0) {
        p->stream << "  WEIGHTS: ";
        for (const auto& pair : joint->weights) {
          p->stream << "\n    " << pair.first << ": " << pair.second;
        }
        p->stream << "\n";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCPrimNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* prim = static_cast<const MSCPrimNode*>(node.get());
      p->PrintIndent();
      MSC_NODE_BASE_HEAD(p->stream, prim, "P");
      p->stream << "  OPTYPE: " << prim->optype;
      if (prim->attrs.size() > 0) {
        p->stream << "\n  ATTRS: ";
        for (const auto& pair : prim->attrs) {
          p->stream << pair.first << "=" << pair.second << " ";
        }
      }
      p->stream << "\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WeightJointNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* joint = static_cast<const WeightJointNode*>(node.get());
      p->PrintIndent();
      MSC_NODE_BASE_HEAD(p->stream, joint, "W");
      if (joint->friends.size() > 0) {
        p->stream << "  FRIENDS: ";
        for (size_t i = 0; i < joint->friends.size(); i++) {
          p->stream << Downcast<BaseJoint>(joint->friends[i])->name
                    << (i == joint->friends.size() - 1 ? "\n" : ",");
        }
      }
      p->stream << "  WEIGHT_TYPE: " << joint->weight_type;
      p->stream << "\n  WEIGHT: " << joint->weight;
      if (joint->attrs.size() > 0) {
        p->stream << "\n  ATTRS: ";
        for (const auto& pair : joint->attrs) {
          p->stream << pair.first << "=" << pair.second << " ";
        }
      }
      p->stream << "\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<WeightGraphNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* graph = static_cast<const WeightGraphNode*>(node.get());
      p->PrintIndent();
      p->stream << graph->name << "\n";
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MSCGraphNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* graph = static_cast<const MSCGraphNode*>(node.get());
      p->PrintIndent();
      p->stream << graph->name << " <INPUTS: ";
      for (size_t i = 0; i < graph->input_names.size(); i++) {
        p->stream << graph->input_names[i] << (i == graph->input_names.size() - 1 ? "| " : ",");
      }
      p->stream << "OUTPUTS: ";
      for (size_t i = 0; i < graph->output_names.size(); i++) {
        p->stream << graph->output_names[i] << (i == graph->output_names.size() - 1 ? ">\n" : ",");
      }
      for (const auto& n : graph->prim_names) {
        p->stream << graph->FindPrim(n) << "\n";
      }
      for (const auto& n : graph->node_names) {
        p->stream << graph->FindNode(n) << "\n";
      }
    });

TVM_REGISTER_NODE_TYPE(MSCTensorNode);

TVM_REGISTER_NODE_TYPE(MSCJointNode);

TVM_REGISTER_NODE_TYPE(MSCPrimNode);

TVM_REGISTER_NODE_TYPE(WeightJointNode);

TVM_REGISTER_NODE_TYPE(MSCGraphNode);

TVM_REGISTER_NODE_TYPE(WeightGraphNode);

TVM_REGISTER_GLOBAL("msc.core.MSCTensor")
    .set_body_typed([](const String& name, const DataType& dtype, const String& layout,
                       const Array<Integer>& shape, const String& alias,
                       const Array<String>& prims) -> MSCTensor {
      return MSCTensor(name, dtype, layout, shape, alias, prims);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorToJson")
    .set_body_typed([](const MSCTensor& tensor) -> String {
      const auto& tensor_json = tensor->ToJson();
      std::ostringstream os;
      dmlc::JSONWriter writer(&os);
      tensor_json.Save(&writer);
      return os.str();
    });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorFromJson")
    .set_body_typed([](const String& tensor_json) -> MSCTensor { return MSCTensor(tensor_json); });

TVM_REGISTER_GLOBAL("msc.core.MSCJoint")
    .set_body_typed([](Integer index, const String& name, const String& shared_ref,
                       const String& optype, const Map<String, String>& attrs,
                       const Array<String>& scope, const Array<MSCJoint>& parents,
                       const Array<Integer> out_indices, const Array<MSCTensor>& outputs,
                       const Map<String, MSCTensor>& weights) -> MSCJoint {
      std::vector<std::pair<BaseJoint, size_t>> inputs;
      for (size_t i = 0; i < parents.size(); i++) {
        inputs.push_back(std::make_pair(parents[i], out_indices[i]->value));
      }
      return MSCJoint(index->value, name, shared_ref, optype, attrs, scope, inputs, outputs,
                      weights);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCPrim")
    .set_body_typed([](Integer index, const String& name, const String& optype,
                       const Map<String, String>& attrs, const Array<MSCPrim>& parents) -> MSCPrim {
      Array<BaseJoint> b_parents;
      for (const auto& p : parents) {
        b_parents.push_back(p);
      }
      return MSCPrim(index->value, name, optype, b_parents, attrs);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightJoint")
    .set_body_typed([](Integer index, const String& name, const String& shared_ref,
                       const String& weight_type, const MSCTensor& weight,
                       const Array<BaseJoint> parents, const Map<String, String>& attrs,
                       const Array<BaseJoint>& friends) -> WeightJoint {
      Array<BaseJoint> b_parents, b_friends;
      for (const auto& p : parents) {
        b_parents.push_back(p);
      }
      for (const auto& f : friends) {
        b_friends.push_back(f);
      }
      return WeightJoint(index->value, name, shared_ref, weight_type, weight, b_parents, attrs,
                         b_friends);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightJointSetAttr")
    .set_body_typed([](const WeightJoint& node, const String& key, const String& value) {
      node->attrs.Set(key, value);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraph")
    .set_body_typed([](const String& name, const Array<MSCJoint>& nodes,
                       const Array<String>& input_names, const Array<String>& output_names,
                       const Array<MSCPrim>& prims) -> MSCGraph {
      return MSCGraph(name, nodes, input_names, output_names, prims);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraph")
    .set_body_typed([](const MSCGraph& graph, const Map<String, Array<String>>& main_wtypes,
                       const Map<String, String>& relation_wtypes) -> WeightGraph {
      return WeightGraph(graph, main_wtypes, relation_wtypes);
    });

// MSC Graph APIS
TVM_REGISTER_GLOBAL("msc.core.MSCGraphHasNode")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> Bool {
      return Bool(graph->HasNode(name));
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindNode")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCJoint {
      return graph->FindNode(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindPrim")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCPrim {
      return graph->FindPrim(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphHasTensor")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> Bool {
      return Bool(graph->HasTensor(name));
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindTensor")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCTensor {
      return graph->FindTensor(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphSetTensorAlias")
    .set_body_typed([](const MSCGraph& graph, const MSCTensor& tensor, const String& alias) {
      tensor->alias = alias;
      graph->tensor_alias.Set(alias, tensor->name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindProducer")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> MSCJoint {
      return graph->FindProducer(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFindConsumers")
    .set_body_typed([](const MSCGraph& graph, const String& name) -> Array<MSCJoint> {
      return graph->FindConsumers(name);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphInputAt")
    .set_body_typed([](const MSCGraph& graph, int index) -> MSCTensor {
      return graph->InputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphOutputAt")
    .set_body_typed([](const MSCGraph& graph, int index) -> MSCTensor {
      return graph->OutputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphGetInputs")
    .set_body_typed([](const MSCGraph& graph) -> Array<MSCTensor> { return graph->GetInputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphGetOutputs")
    .set_body_typed([](const MSCGraph& graph) -> Array<MSCTensor> { return graph->GetOutputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphToJson").set_body_typed([](const MSCGraph& graph) -> String {
  const auto& graph_json = graph->ToJson();
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  graph_json.Save(&writer);
  return os.str();
});

TVM_REGISTER_GLOBAL("msc.core.MSCGraphFromJson")
    .set_body_typed([](const String& graph_json) -> MSCGraph { return MSCGraph(graph_json); });

TVM_REGISTER_GLOBAL("msc.core.MSCGraphToPrototxt")
    .set_body_typed([](const MSCGraph& graph) -> String { return graph->ToPrototxt(); });

// Weight Graph APIS
TVM_REGISTER_GLOBAL("msc.core.WeightGraphHasNode")
    .set_body_typed([](const WeightGraph& graph, const String& name) -> Bool {
      return Bool(graph->HasNode(name));
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraphFindNode")
    .set_body_typed([](const WeightGraph& graph, const String& name) -> WeightJoint {
      return graph->FindNode(name);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraphToJson")
    .set_body_typed([](const WeightGraph& graph) -> String {
      const auto& graph_json = graph->ToJson();
      std::ostringstream os;
      dmlc::JSONWriter writer(&os);
      graph_json.Save(&writer);
      return os.str();
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraphFromJson")
    .set_body_typed([](const String& graph_json) -> WeightGraph {
      return WeightGraph(graph_json);
    });

TVM_REGISTER_GLOBAL("msc.core.WeightGraphToPrototxt")
    .set_body_typed([](const WeightGraph& graph) -> String { return graph->ToPrototxt(); });

TVM_REGISTER_GLOBAL("msc.core.MSCJointInputAt")
    .set_body_typed([](const MSCJoint& node, int index) -> MSCTensor {
      return node->InputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointOutputAt")
    .set_body_typed([](const MSCJoint& node, int index) -> MSCTensor {
      return node->OutputAt(index);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointWeightAt")
    .set_body_typed([](const MSCJoint& node, const String& wtype) -> MSCTensor {
      return node->WeightAt(wtype);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetInputs")
    .set_body_typed([](const MSCJoint& node) -> Array<MSCTensor> { return node->GetInputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetOutputs")
    .set_body_typed([](const MSCJoint& node) -> Array<MSCTensor> { return node->GetOutputs(); });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetWeights")
    .set_body_typed([](const MSCJoint& node) -> Map<String, MSCTensor> { return node->weights; });

TVM_REGISTER_GLOBAL("msc.core.MSCJointHasAttr")
    .set_body_typed([](const MSCJoint& node, const String& key) -> Bool {
      return Bool(node->HasAttr(key));
    });

TVM_REGISTER_GLOBAL("msc.core.MSCJointGetAttrs")
    .set_body_typed([](const MSCJoint& node) -> Map<String, String> { return node->attrs; });

TVM_REGISTER_GLOBAL("msc.core.WeightJointHasAttr")
    .set_body_typed([](const WeightJoint& node, const String& key) -> Bool {
      return Bool(node->HasAttr(key));
    });

TVM_REGISTER_GLOBAL("msc.core.WeightJointGetAttrs")
    .set_body_typed([](const WeightJoint& node) -> Map<String, String> { return node->attrs; });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorDTypeName")
    .set_body_typed([](const MSCTensor& tensor) -> String { return tensor->DTypeName(); });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorDimAt")
    .set_body_typed([](const MSCTensor& tensor, const String& axis) -> Integer {
      return tensor->DimAt(axis);
    });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorGetSize")
    .set_body_typed([](const MSCTensor& tensor) -> Integer { return tensor->GetSize(); });

TVM_REGISTER_GLOBAL("msc.core.MSCTensorSetAlias")
    .set_body_typed([](const MSCTensor& tensor, const String& alias) { tensor->alias = alias; });

TVM_REGISTER_GLOBAL("msc.core.PruneWeights")
    .set_body_typed([](const MSCGraph& graph,
                       const Map<String, MSCTensor>& pruned_tensors) -> MSCGraph {
      return PruneWeights(graph, pruned_tensors);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
