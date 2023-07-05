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
 * \file ad_utils.cc
 * \brief Utility for tensor-level auto-differentiation.
 */
#include "ad_utils.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <set>
#include <string>

#include "../schedule/operation_inline.h"

namespace tvm {
namespace te {

std::pair<Array<IterVar>, Map<Var, PrimExpr>> CloneIterVars(const Array<IterVar>& vars) {
  Array<IterVar> new_vars;
  Map<Var, PrimExpr> vmap;
  for (const IterVar& iv : vars) {
    IterVar new_v = IterVar(iv->dom, iv->var.copy_with_suffix(""), iv->iter_type, iv->thread_tag);
    new_vars.push_back(new_v);
    vmap.Set(iv->var, new_v->var);
  }
  return std::make_pair(std::move(new_vars), std::move(vmap));
}

PrimExpr CloneReduction(const PrimExpr& expr) {
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    auto [new_axis, vmap] = CloneIterVars(red->axis);

    Array<PrimExpr> src_with_newaxis;
    for (const auto& src : red->source) {
      src_with_newaxis.push_back(tir::Substitute(src, vmap));
    }
    Array<PrimExpr> init_with_newaxis;
    for (const auto& init : red->init) {
      init_with_newaxis.push_back(tir::Substitute(init, vmap));
    }

    return Reduce(red->combiner, src_with_newaxis, new_axis, tir::Substitute(red->condition, vmap),
                  red->value_index, init_with_newaxis);
  } else {
    return expr;
  }
}

Operation ComputeOpFromExprs(const Array<PrimExpr>& exprs, const Array<IterVar>& axis,
                             const std::string& name, const std::string& tag,
                             const Map<String, ObjectRef>& attrs, bool clone_axis) {
  if (clone_axis) {
    auto [new_axis, vmap] = CloneIterVars(axis);
    Array<PrimExpr> new_exprs;
    for (const PrimExpr& e : exprs) {
      new_exprs.push_back(Substitute(CloneReduction(e), vmap));
    }
    return ComputeOpFromExprs(new_exprs, new_axis, name, tag, attrs, false);
  }

  Array<PrimExpr> new_exprs;

  // If this is a reduction then we have to replicate it
  if (const ReduceNode* red = exprs[0].as<ReduceNode>()) {
    for (size_t i = 0; i < red->source.size(); ++i) {
      PrimExpr ith_red =
          Reduce(red->combiner, red->source, red->axis, red->condition, i, red->init);
      new_exprs.push_back(ith_red);
    }
  } else {
    new_exprs = exprs;
  }

  return ComputeOp(name, tag, attrs, axis, new_exprs);
}

Tensor TensorFromExpr(const PrimExpr& expr, const Array<IterVar>& axis, const std::string& name,
                      const std::string& tag, const Map<String, ObjectRef>& attrs,
                      bool clone_axis) {
  int new_value_index = 0;
  if (const ReduceNode* red = expr.as<ReduceNode>()) {
    new_value_index = red->value_index;
  }
  return ComputeOpFromExprs({expr}, axis, name, tag, attrs, clone_axis).output(new_value_index);
}

Tensor TransformTensorBody(
    const Tensor& tensor,
    const std::function<PrimExpr(const PrimExpr&, const Array<IterVar>&)>& func) {
  if (const ComputeOpNode* op = tensor->op.as<ComputeOpNode>()) {
    // Transform only one body
    PrimExpr new_body = func(op->body[tensor->value_index], op->axis);

    // If the body didn't change then we can return the same tensor
    if (new_body.same_as(op->body[tensor->value_index])) {
      return tensor;
    }

    return TensorFromExpr(new_body, op->axis, op->name, op->tag, op->attrs);
  } else {
    return tensor;
  }
}

Tensor TransformTensorBody(const Tensor& tensor,
                           const std::function<PrimExpr(const PrimExpr&)>& func) {
  return TransformTensorBody(tensor,
                             [func](const PrimExpr& e, const Array<IterVar>&) { return func(e); });
}

// If expr is a Tensor Access node, perform inlining, otherwise do nothing
PrimExpr InlineImmediateTensorAccess(const PrimExpr& expr) {
  if (const ProducerLoadNode* op = expr.as<ProducerLoadNode>()) {
    auto tensor = Downcast<te::Tensor>(op->producer);
    if (const ComputeOpNode* op_comp = tensor->op.as<ComputeOpNode>()) {
      Array<Var> tensor_axes;
      for (const auto& var : op_comp->axis) {
        tensor_axes.push_back(var->var);
      }

      Stmt inlined =
          Inline(Evaluate(expr), tensor->op, tensor_axes, op_comp->body[tensor->value_index]);
      if (const EvaluateNode* ev = inlined.as<EvaluateNode>()) {
        // If it is a reduction, clone it
        return CloneReduction(ev->value);
      }
    }
  }
  return expr;
}

// Implements InlineTensors by trying to inline every Call of the given Expr
class InlineTensorsMutator : public ExprMutator {
 public:
  explicit InlineTensorsMutator(const Array<Tensor>& inlineable, bool inline_reductions = false)
      : inline_reductions_(inline_reductions) {
    for (const Tensor& tensor : inlineable) {
      inlineable_.emplace(tensor->op.operator->(), tensor->value_index);
    }
  }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);
    if (const ComputeOpNode* op_comp = tensor->op.as<ComputeOpNode>()) {
      // Inline only if the array of inlineable tensors is empty or contains this tensor
      if (inlineable_.empty() || inlineable_.count({op_comp, tensor->value_index})) {
        // Inline only compute nodes that are not reductions (unless inline reductions is allowed)
        if (inline_reductions_ || !op_comp->body[0].as<ReduceNode>()) {
          PrimExpr expr = GetRef<PrimExpr>(op);
          // Inline this tensor access and then try to perform further inlining
          return VisitExpr(InlineImmediateTensorAccess(expr));
        }
      }
    }
    // If we cannot inline this call, we should try to do inlining in its arguments
    return ExprMutator::VisitExpr_(op);
  }

 private:
  // Tensors which are allowed to be inlined, represented as pairs (op_node, value_index)
  std::set<std::pair<const OperationNode*, int>> inlineable_;
  bool inline_reductions_;
};

Tensor InlineTensorAccess(const Tensor& tensor, const Array<Tensor>& inlineable,
                          bool inline_reductions) {
  auto transformation = [inlineable, inline_reductions](const PrimExpr& e) {
    return InlineTensorsMutator(inlineable, inline_reductions)(e);
  };
  return TransformTensorBody(tensor, transformation);
}

Tensor InlineTailTensorAccess(const Tensor& tensor) {
  return TransformTensorBody(tensor, InlineImmediateTensorAccess);
}

}  // namespace te
}  // namespace tvm
