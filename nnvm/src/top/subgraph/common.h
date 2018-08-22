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

#ifndef NNVM_TOP_SUBGRAPH_COMMON_H_
#define NNVM_TOP_SUBGRAPH_COMMON_H_

#include <topi/nn.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/top/tensor.h>
#include <string>
#include <set>
#include <vector>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

inline uint32_t DefaultSubgraphOpNumInputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListInputNames(nnvm::Symbol::kAll).size();
}

inline uint32_t DefaultSubgraphOpNumOutputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListOutputNames().size();
}

inline std::vector<std::string> DefaultSubgraphOpListInputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListInputNames(nnvm::Symbol::kAll);
}

inline std::vector<std::string> DefaultSubgraphOpListOutputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& sym = *attrs.subgraphs[0];
  return sym.ListOutputNames();
}

inline bool DefaultSubgraphOpShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape> *in_shapes,
                                   std::vector<TShape> *out_shapes) {
  const nnvm::Symbol& subgraph_sym = *attrs.subgraphs[0];
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_shapes->size());
  CHECK_EQ(idx_g.outputs().size(), out_shapes->size());

  // Put the input and output shapes to the shape vector.
  nnvm::ShapeVector shapes(idx_g.num_node_entries());
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_shapes->size());
  for (size_t i = 0; i < in_shapes->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    shapes[eid] = in_shapes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_shapes->size());
  for (size_t i = 0; i < out_shapes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    shapes[eid] = out_shapes->at(i);
  }

  // Infer shape of the graph.
  g.attrs["shape"] = std::make_shared<dmlc::any>(std::move(shapes));
  // g = exec::InferShape(std::move(g));
  g = ApplyPass(std::move(g), "InferShape");

  // Copy the inferred shape back to the input shapes and the output shapes.
  shapes = g.GetAttr<nnvm::ShapeVector>("shape");
  // assign to in_shapes
  for (size_t i = 0; i < in_shapes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    // SHAPE_ASSIGN_CHECK(*in_shapes, i, shapes[eid]);
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shapes, i, shapes[eid]);
  }
  // assign to out_shapes
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    // SHAPE_ASSIGN_CHECK(*out_shapes, i, shapes[eid]);
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shapes, i, shapes[eid]);
  }
  // Check if we have inferred the shapes correctly.
  return g.GetAttr<size_t>("shape_num_unknown_nodes") == 0;
}

inline bool DefaultSubgraphOpType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_types,
                                  std::vector<int> *out_types) {
  const nnvm::Symbol& subgraph_sym = *attrs.subgraphs[0];
  nnvm::Graph g;
  g.outputs = subgraph_sym.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());

  // Put the input and output data types to the dtype vector.
  nnvm::DTypeVector types(idx_g.num_node_entries(), -1);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_types->size());
  for (size_t i = 0; i < in_types->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    types[eid] = in_types->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_types->size());
  for (size_t i = 0; i < out_types->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    types[eid] = out_types->at(i);
  }

  // Infer data type of the graph.
  g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
  // g = exec::InferType(std::move(g));
  g = ApplyPass(std::move(g), "InferType");

  types = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < in_types->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    // TYPE_ASSIGN_CHECK(*in_types, i, types[eid]);
    NNVM_ASSIGN_INPUT_TYPE(attrs, *in_types, i, types[eid]);
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    // TYPE_ASSIGN_CHECK(*out_types, i, types[eid]);
    NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_types, i, types[eid]);
  }
  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

// TODO(junwu): Implement this function
inline bool CorrectLayout(const NodeAttrs& attrs,
                          std::vector<Layout> *ilayouts,
                          const std::vector<Layout> *last_ilayouts,
                          std::vector<Layout> *olayouts) {
  return true;
}

inline std::vector<uint32_t> DefaultSubgraphOpMutableInputs(const nnvm::NodeAttrs& attrs) {
  const nnvm::Symbol& subgraph_sym = *attrs.subgraphs[0];
  const std::vector<std::string> input_names = subgraph_sym.ListInputNames(nnvm::Symbol::kAll);
  const std::vector<std::string> immutable_input_names =
    subgraph_sym.ListInputNames(nnvm::Symbol::kReadOnlyArgs);
  const std::vector<std::string> mutable_input_names =
    subgraph_sym.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
  CHECK_EQ(immutable_input_names.size() + mutable_input_names.size(), input_names.size());
  std::vector<uint32_t> ret;
  size_t i1 = 0, i2 = 0;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i1 < immutable_input_names.size() && input_names[i] == immutable_input_names[i1]) {
      ++i1;
    } else {
      CHECK(i2 < mutable_input_names.size());
      CHECK_EQ(input_names[i], mutable_input_names[i2]);
      ++i2;
      ret.push_back(i);
    }
  }
  return ret;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_SUBGRAPH_COMMON_H_
