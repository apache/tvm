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
 * \file linear_system.cc
 * \brief The linear system data structures.
 */
#include <tvm/arith/linear_system.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/runtime/registry.h>

#include <utility>
#include <algorithm>
#include <unordered_map>

namespace tvm {
namespace arith {

LinearSystem::LinearSystem(Array<Var> variables,
                           Map<Var, Range> ranges,
                           Array<PrimExpr> relations) {
  ObjectPtr<LinearSystemNode> node = make_object<LinearSystemNode>();
  node->variables = std::move(variables);
  node->ranges = std::move(ranges);
  node->relations = std::move(relations);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(LinearSystemNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LinearSystemNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LinearSystemNode*>(node.get());
    p->stream << "LinearSystem("
              << op->variables
              << ", " << op->ranges
              << ", " << op->relations
              << ")";
  });


LinearSystemTransform::LinearSystemTransform(LinearSystem src,
                                             LinearSystem dst,
                                             Map<Var, PrimExpr> src_to_dst,
                                             Map<Var, PrimExpr> dst_to_src) {
  ObjectPtr<LinearSystemTransformNode> node = make_object<LinearSystemTransformNode>();
  node->src = std::move(src);
  node->dst = std::move(dst);
  node->src_to_dst = std::move(src_to_dst);
  node->dst_to_src = std::move(dst_to_src);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(LinearSystemTransformNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LinearSystemTransformNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const LinearSystemTransformNode*>(node.get());
    p->stream << "LinearSystemTransform("
              << "\n\t" << op->src
              << "\n\t" << op->dst
              << "\n\t" << op->src_to_dst
              << "\n\t" << op->dst_to_src
              << "\n)";
  });

}  // namespace arith
}  // namespace tvm
