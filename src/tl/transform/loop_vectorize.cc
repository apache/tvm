/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file loop_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "loop_vectorize.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../../arith/int_operator.h"
#include "../../arith/ir_visitor_with_analyzer.h"
#include "../layout/layout.h"
#include "../layout/utils.h"
#include "common/loop_vectorization_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class VectorizePlanner : public arith::IRVisitorWithAnalyzer {
 public:
  VectorizePlanner() = default;

  int Plan(const For& node) {
    this->operator()(node);
    // Always Enable vectorization
    // if (!has_nonlocal_memory_access_) return 1;
    return vector_size_;
  }

  bool GetDynamic() { return dynamic_; }

  PrimExpr GetCondition() { return condition_; }

 private:
  void VisitStmt_(const ForNode* node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));
    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode* node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1 && node->buffer->shape[0].as<IntImmNode>()->value == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
    }
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void VisitStmt_(const BufferStoreNode* node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitStmt_(const IfThenElseNode* node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode* node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr& cond) {
    // TODO: perform some checks here
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer& buffer) {
    if (!inner_for_) return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr) return;

    const DataType& access_type = buffer->dtype;
    // i // 2, i % 8 can also be vectorized as factor 16
    int max_vector_size = 128 / access_type.bits();
    // so we should disable this GCD optimization
    max_vector_size = arith::ZeroAwareGCD(max_vector_size, extent_ptr->value);

    auto mod_set = analyzer_.modular_set(buffer->shape.back());
    // when dynamic shape like [m, k]: coeff=1, base=0, GCD will block conditionally tail vectorize
    if (buffer->shape.back().as<IntImmNode>()) {
      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);

      // comment as this solution doesn't
      // work well with multi-index vectorization
      // max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->base);

      vector_size_ = arith::ZeroAwareGCD(max_vector_size, vector_size_);

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        elem_offset = elem_offset + indices[i] * stride;
        stride = stride * buffer->shape[i];
      }
      while (!IndiceCanVectorize(elem_offset, inner_for_->loop_var, inner_for_->extent,
                                 vector_size_, &analyzer_)) {
        vector_size_ /= 2;
      }
    } else if (vector_size_ <= vector_load_bits_max_ / buffer->dtype.bits()) {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      PrimExpr offset = buffer.OffsetOf(indices).back();
      condition_ = (FloorMod(offset, vector_size_) == 0);
    }
  }

  static const int vector_load_bits_max_ = 128;

  const ForNode* inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  // conditionally vectorize
  bool dynamic_ = false;
  PrimExpr condition_;
};

class VectorizeDynamicCallRemover : public StmtExprMutator {
 public:
  VectorizeDynamicCallRemover(Var inner_var, int vector_size)
      : inner_var_(inner_var), vector_size_(vector_size) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr cond = this->VisitExpr(op->args[0]);
      Map<Var, PrimExpr> vmap;
      // Currently remove upper bound check
      vmap.Set(inner_var_, 0);
      cond = Substitute(cond, vmap);
      Array<PrimExpr> new_args{cond, op->args[1], op->args[2]};
      return Call(op->dtype, op->op, new_args, op->span);
    } else {
      // TODO: For other calls
      return GetRef<PrimExpr>(op);
    }
  }

  Var inner_var_;
  int vector_size_;
};

class VectorizeRewriter : public StmtExprMutator {
 public:
  VectorizeRewriter(VectorizePlanResult plan)
      : vector_size_(plan.vector_size), condition_(plan.condition), dynamic_(plan.dynamic) {}

 private:
  Stmt VisitStmt_(const ForNode* node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) {  // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_;
      ICHECK(is_zero(fnode->min));
      if (!dynamic_) {  // check dynamic shape
        if (extent == vector_size_) {
          fnode.CopyOnWrite()->kind = ForKind::kVectorized;
          return fnode;
        } else {
          Var inner_var = Var("vec");
          Var outer_var = Var(old_var->name_hint);
          Map<Var, PrimExpr> vmap;
          vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
          Stmt body = Substitute(fnode->body, vmap);
          body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
          body = For(outer_var, 0, extent / vector_size_, fnode->kind, body, fnode->thread_binding,
                     fnode->annotations, fnode->span);
          return body;
        }
      } else {
        Var inner_var = Var("vec");
        Var outer_var = Var(old_var->name_hint);
        Map<Var, PrimExpr> vmap;
        vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
        Stmt body = Substitute(fnode->body, vmap);
        // add condition ifthenelse here
        Map<Var, PrimExpr> vmap_condition;
        vmap_condition.Set(fnode->loop_var, outer_var * vector_size_);
        PrimExpr condition = Substitute(condition_, vmap_condition);

        VectorizeDynamicCallRemover remover(inner_var, vector_size_);
        body = remover(body);

        For vectorize_for = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
        For serial_for = For(inner_var, 0, vector_size_, ForKind::kSerial, body);
        body = IfThenElse(condition, vectorize_for, serial_for);
        body = For(outer_var, 0, extent / vector_size_, fnode->kind, body, fnode->thread_binding,
                   fnode->annotations, fnode->span);
        return body;
      }
    } else {
      return ret;
    }
  }

  const ForNode* inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
};

int GetVectorizeSize(const For& loop) { return VectorizePlanner().Plan(loop); }

VectorizePlanResult GetVectorizePlanResult(const For& loop) {
  VectorizePlanner planner;
  int vector_size = planner.Plan(loop);
  bool dynamic = planner.GetDynamic();
  PrimExpr condition = planner.GetCondition();
  return {vector_size, dynamic, condition};
}

bool IndiceCanVectorize(PrimExpr expr, Var var, PrimExpr iter_var_size, int target_vectorized_size,
                        arith::Analyzer* analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1) return true;
  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_vectorized_size), 0)) return false;
  Var v0("v0"), v1("v1");
  analyzer->Bind(v0, Range(0, target_vectorized_size));
  analyzer->Bind(v1, Range(0, FloorDiv(iter_var_size, target_vectorized_size)));
  PrimExpr expr_transformed =
      analyzer->Simplify(Substitute(expr, {{var, v0 + v1 * target_vectorized_size}}));

  Vectorizer vectorizer(v0, IntImm(v0->dtype, target_vectorized_size));
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    else
      return false;
  } else {
    return is_one(ramp_node->stride);
  }
}

For VectorizeLoop(const For& loop, int vectorize_hint) {
  VectorizePlanResult res{128, false, 0};
  if (vectorize_hint <= 0) {
    res = GetVectorizePlanResult(loop);
    vectorize_hint = res.vector_size;
  }
  if (vectorize_hint == 1) return loop;
  auto rewriter = VectorizeRewriter(res);
  return Downcast<For>(rewriter(loop));
}

}  // namespace tl
}  // namespace tvm
