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
  * \file frontend_legalize.cc
  * \brief Legalize the program from frontend
  */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/op.h>

#include "../arith/ir_mutator_with_analyzer.h"
#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class FrontendLegalizer : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    FrontendLegalizer substituter(&analyzer);
    for (const auto& [_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode* op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* node) final {
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_++;
    }
    auto n = StmtExprMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) {
      parallel_for_scope_--;
    }
    return n;
  }

  Stmt VisitStmt_(const EvaluateNode* node) final {
    if (auto call = node->value.as<CallNode>()) {
      if (call->op.same_as(tl::fill())) {
        return LowerFill(call->args);
      } else if (call->op.same_as(tl::copy())) {
        return LowerCopy(call->args);
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt LowerCopy(const Array<PrimExpr>& call_args) {
    CopyArgs args = CopyArgs::Parse(call_args);
    Array<IterVar> loop_vars = args.MakeIterVars();
    for (const auto& iv : loop_vars) analyzer_->Bind(iv->var, iv->dom);

    Array<PrimExpr> src_indices = args.MakeIndices(loop_vars, 0);
    Array<PrimExpr> dst_indices = args.MakeIndices(loop_vars, 1);

    PrimExpr src_predicate = args.MakePredicate(analyzer_, loop_vars, args.src->shape, 0);
    PrimExpr dst_predicate = args.MakePredicate(analyzer_, loop_vars, args.dst->shape, 1);

    PrimExpr value = BufferLoad(args.src, src_indices);
    if (args.src->dtype != args.dst->dtype) value = Cast(args.dst->dtype, value);
    if (src_predicate.defined()) value = if_then_else(src_predicate, value, make_zero(args.dst->dtype));

    Stmt body = BufferStore(args.dst, value, dst_indices);
    if (dst_predicate.defined()) body = IfThenElse(dst_predicate, body);

    for (int i = loop_vars.size() - 1; i >= 0; i--) {
      body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent, ForKind::kParallel, body);
    }
    return body;
  }

  Stmt LowerFill(const Array<PrimExpr>& clear_args) {
    FillArgs args = FillArgs::Parse(clear_args, buffer_data_to_buffer_);
    int ndim = args.dst->shape.size();
    Array<IterVar> loop_vars;
    Array<PrimExpr> dst_indices;
    for (int i = 0; i < ndim; i++) {
      Var var = Var(std::string{ char('i' + i) });
      loop_vars.push_back({ Range(0, args.dst->shape[i]), var, IterVarType::kDataPar });
      dst_indices.push_back(var);
    }
    Stmt body = BufferStore(args.dst, args.value, dst_indices);
    for (int i = ndim - 1; i >= 0; i--) {
      body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent, ForKind::kParallel, body);
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* node) final {
    if (let_bindings_.count(node)) {
      return arith::IRMutatorWithAnalyzer::VisitExpr(let_bindings_[node]);
    } else {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* node) final {
    if (parallel_for_scope_ > 0) {
      let_bindings_[node->var.get()] = node->value;
      return arith::IRMutatorWithAnalyzer::VisitStmt(node->body);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const LetNode* node) final {
    if (parallel_for_scope_ > 0) {
      let_bindings_[node->var.get()] = node->value;
      return arith::IRMutatorWithAnalyzer::VisitExpr(node->body);
    }
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  int parallel_for_scope_ = 0;
  std::unordered_map<const VarNode*, PrimExpr> let_bindings_;
  Map<Var, Buffer> buffer_data_to_buffer_;
};

using namespace tir::transform;

Pass FrontendLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FrontendLegalizer::Substitute(std::move(f));
    };
  return CreatePrimFuncPass(pass_func, 0, "tl.FrontendLegalize", {});
}

TVM_REGISTER_GLOBAL("tl.FrontendLegalize").set_body_typed(FrontendLegalize);

} // namespace tl
} // namespace tvm
