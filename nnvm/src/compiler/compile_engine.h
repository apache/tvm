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
 *  Copyright (c) 2017 by Contributors
 * \file compile_engine.h
 * \brief Internal engine to compile a subgraph fragment and cache compilation.
 */
#ifndef NNVM_COMPILER_COMPILE_ENGINE_H_
#define NNVM_COMPILER_COMPILE_ENGINE_H_

#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/operation.h>
#include <tvm/lowered_func.h>
#include <string>
#include <utility>
#include "graph_hash.h"

namespace nnvm {
namespace compiler {

/*! \brief A TVM Node to represent compiled graph function */
struct GraphFuncNode : public tvm::Node {
  /* \brief compiled target */
  std::string target;
  /*! \brief Function name */
  std::string func_name;
  /* \brief The inputs to the function */
  tvm::Array<Tensor> inputs;
  /* \brief The outputs to the function */
  tvm::Array<Tensor> outputs;
  /*! \brief The lowered functions */
  tvm::Array<tvm::LoweredFunc> funcs;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("target", &target);
    v->Visit("func_name", &func_name);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("funcs", &funcs);
  }

  static constexpr const char* _type_key = "GraphFunc";
  TVM_DECLARE_NODE_TYPE_INFO(GraphFuncNode, tvm::Node);
};

TVM_DEFINE_NODE_REF(GraphFunc, GraphFuncNode);

/*! \brief Cache Entry in the graph */
struct GraphCacheEntryNode : public tvm::Node {
  /*! \brief The graph function */
  GraphFunc graph_func;
  /*! \brief Usage statistics */
  int use_count{0};
  /*! \brief Index of the master node for calling schedule*/
  int master_idx;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("graph_func", &graph_func);
    v->Visit("use_count", &use_count);
    v->Visit("master_idx", &master_idx);
  }
  static constexpr const char* _type_key = "GraphCacheEntry";
  TVM_DECLARE_NODE_TYPE_INFO(GraphCacheEntryNode, tvm::Node);
};

class GraphCacheEntry : public ::tvm::NodeRef {
 public:
  GraphCacheEntry() {}
  explicit GraphCacheEntry(::tvm::NodePtr<::tvm::Node> n) : NodeRef(n) {}
  GraphCacheEntryNode* operator->() {
    return static_cast<GraphCacheEntryNode*>(node_.get());
  }
  using ContainerType = GraphCacheEntryNode;
};

/*!
 * \brief Call compile engine to lower a graph with given inputs.
 *
 * \param graph The graph to be compiled
 * \param inputs The input specification.
 * \param target The build target
 * \param master_idx The index of master node for calling schedule
 *
 * \return func A lowered tvm function.
 */
GraphFunc GraphLower(Graph graph,
                     const Array<tvm::Tensor>& inputs,
                     const std::string& target,
                     int master_idx);

/*!
 * \brief Get type flag from TVM Type
 *
 * \param type the tvm type
 * \return corresponding DLDataType
 */
int GetTypeFlag(tvm::Type type);

/*!
 * \brief Get TVM Type from type flag
 *
 * \param type_flag the type flag
 * \return corresponding TVM type
 */
tvm::Type GetTVMType(int type_flag);

}  // namespace compiler
}  // namespace nnvm

#endif  // NNVM_COMPILER_COMPILE_ENGINE_H_
