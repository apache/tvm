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
 * \file int_constraints.cc
 * \brief The integer constraints data structures.
 */
#include <tvm/arith/int_solver.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/runtime/registry.h>

#include <utility>
#include <algorithm>
#include <unordered_map>

namespace tvm {
namespace arith {

IntConstraints::IntConstraints(Array<Var> variables,
                               Map<Var, Range> ranges,
                               Array<PrimExpr> relations) {
  ObjectPtr<IntConstraintsNode> node = make_object<IntConstraintsNode>();
  if (!variables.defined()) {
    variables = Array<Var>();
  }
  if (!ranges.defined()) {
    ranges = Map<Var, Range>();
  }
  CHECK(relations.defined());
  for (const auto& var : variables) {
    CHECK(var.dtype().is_int() || var.dtype().is_uint())
      << "Variables in IntConstraints must be integers";
  }
  node->variables = std::move(variables);
  node->ranges = std::move(ranges);
  node->relations = std::move(relations);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IntConstraintsNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<IntConstraintsNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const IntConstraintsNode*>(node.get());
    p->stream << "IntConstraints("
              << op->variables
              << ", " << op->ranges
              << ", " << op->relations
              << ")";
  });


IntConstraintsTransform::IntConstraintsTransform(IntConstraints src,
                                                 IntConstraints dst,
                                                 Map<Var, PrimExpr> src_to_dst,
                                                 Map<Var, PrimExpr> dst_to_src) {
  ObjectPtr<IntConstraintsTransformNode> node = make_object<IntConstraintsTransformNode>();
  node->src = std::move(src);
  node->dst = std::move(dst);
  node->src_to_dst = std::move(src_to_dst);
  node->dst_to_src = std::move(dst_to_src);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IntConstraintsTransformNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<IntConstraintsTransformNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* op = static_cast<const IntConstraintsTransformNode*>(node.get());
    p->stream << "IntConstraintsTransform("
              << "\n\t" << op->src
              << "\n\t" << op->dst
              << "\n\t" << op->src_to_dst
              << "\n\t" << op->dst_to_src
              << "\n)";
  });

}  // namespace arith
}  // namespace tvm
