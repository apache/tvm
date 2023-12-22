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
 * \file auto_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "auto_vectorize.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../arith/int_operator.h"
#include "../arith/ir_visitor_with_analyzer.h"
#include "helper.h"
#include "layout.h"

namespace tvm {
namespace tl {

using namespace tir;

class VectorizePlanner : public arith::IRVisitorWithAnalyzer {
 public:
  VectorizePlanner() = default;

  int Plan(const For& node) {
    this->operator()(node);
    if (!has_nonlocal_memory_access_) return 1;
    return vector_size_;
  }

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
    int max_vector_size = arith::ZeroAwareGCD(128 / access_type.bits(), extent_ptr->value);

    auto buffer_last_dim_size = as_const_int(buffer->shape.back());
    ICHECK(buffer_last_dim_size != nullptr)
        << "dyn shape currently not supported " << buffer->shape;
    max_vector_size = arith::ZeroAwareGCD(max_vector_size, *buffer_last_dim_size);

    PrimExpr lsi = indices.back();
    auto iter_sum = arith::NormalizeToIterSum(lsi, iter_map_, &analyzer_);
    int access_vector_size = GetVectorSize(iter_sum, inner_for_->loop_var, max_vector_size);
    int vector_size = arith::ZeroAwareGCD(max_vector_size, access_vector_size);
    vector_size_ = arith::ZeroAwareGCD(vector_size, vector_size_);
  }

  int GetVectorSize(arith::IterSumExpr iter_sum, Var last_var, int max_vector_size) {
    int vector_size = 2;
    while ((max_vector_size % vector_size) == 0) {
      bool can_vector_load = true;
      if (!analyzer_.CanProveEqual(FloorMod(iter_sum->base, vector_size), 0))
        can_vector_load = false;

      for (const auto& split : iter_sum->args) {
        int scale = split->scale.as<IntImm>().value()->value;
        int lower_factor = split->lower_factor.as<IntImm>().value()->value;
        if (split->source->source.same_as(last_var) && lower_factor % vector_size != 0) {
          if (lower_factor != scale) {
            can_vector_load = false;
            break;
          }
        } else {
          int scale = split->scale.as<IntImm>().value()->value;
          if ((scale % vector_size) != 0) {
            can_vector_load = false;
            break;
          }
        }
      }
      if (!can_vector_load) break;
      vector_size *= 2;
    }
    return vector_size / 2;
  }

  static const int vector_load_bits_max_ = 128;

  const ForNode* inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
};

class VectorizeRewriter : public StmtExprMutator {
 public:
  VectorizeRewriter(int vector_size) : vector_size_(vector_size) {}

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
      ICHECK(extent % vector_size_ == 0);
      ICHECK(is_zero(fnode->min));
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
      return ret;
    }
  }

  const ForNode* inner_for_;
  const int vector_size_;
};

int GetVectorizeSize(const For& loop) { return VectorizePlanner().Plan(loop); }

Stmt VectorizeLoop(const For& loop, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    vectorize_hint = GetVectorizeSize(loop);
  }
  if (vectorize_hint == 1) return loop;
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return rewriter(loop);
}

}  // namespace tl
}  // namespace tvm
