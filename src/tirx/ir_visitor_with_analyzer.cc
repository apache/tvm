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
 * \file tirx/ir_visitor_with_analyzer.cc
 */
#include "ir_visitor_with_analyzer.h"

#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const ForNode* op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    if (auto result = this->Visit(op->min)) return result;
    if (auto result = this->Visit(op->extent)) return result;
    if (op->step.has_value()) {
      if (auto result = this->Visit(*op->step)) return result;
    }
    return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
      constraint_scope_.Current().Emplace(analyzer_, op->extent > IntImm(op->extent.ty(), 0));
      return this->Visit(op->body);
    });
  });
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const SBlockNode* op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    for (const auto& iter_var : op->iter_vars) {
      analyzer_->Bind(iter_var->var, iter_var->dom);
    }
    return StmtExprVisitor::Visit_(op);
  });
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const BindNode* op) {
  if (auto result = this->Visit(op->value)) return result;
  if (ffi::Optional<PrimExpr> value = op->value.as<PrimExpr>()) {
    analyzer_->Bind(op->var, value.value());
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const IfThenElseNode* op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    if (auto result = this->Visit(op->condition)) return result;

    PrimExpr real_condition = ExtractRealCondition(op->condition);

    if (auto result = constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
          constraint_scope_.Current().Emplace(analyzer_, real_condition);
          return this->Visit(op->then_case);
        }))
      return result;
    if (op->else_case) {
      if (auto result = constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
            constraint_scope_.Current().Emplace(
                analyzer_, analyzer_->rewrite_simplify(prim::Not(real_condition)));
            return this->Visit(op->else_case.value());
          }))
        return result;
    }
    return std::nullopt;
  });
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const AttrStmtNode* op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    if (op->attr_key == tirx::attr::thread_extent || op->attr_key == s_tir::attr::virtual_thread) {
      IterVar iv = op->node.as_or_throw<IterVar>();
      TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
      analyzer_->Bind(iv->var, Range::FromMinExtent(IntImm(op->value.ty(), 0), op->value));
    }
    return StmtExprVisitor::Visit_(op);
  });
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const AssertStmtNode* op) {
  if (auto result = this->Visit(op->condition)) return result;
  constraint_scope_.Current().Emplace(analyzer_, op->condition);
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const CallNode* op) {
  // add condition context to if_then_else
  static const Op& if_then_else_op = Op::Get("ir.prim.if_then_else");
  if (op->op.same_as(if_then_else_op)) {
    PrimExpr cond = op->args[0].as_or_throw<PrimExpr>();
    if (auto result = this->Visit(cond)) return result;
    if (auto result = constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
          constraint_scope_.Current().Emplace(analyzer_, cond);
          return this->Visit(op->args[1]);
        }))
      return result;
    if (auto result = constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
          constraint_scope_.Current().Emplace(analyzer_,
                                              analyzer_->rewrite_simplify(prim::Not(cond)));
          return this->Visit(op->args[2]);
        }))
      return result;
  } else {
    if (auto result = StmtExprVisitor::Visit_(op)) return result;
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> IRVisitorWithAnalyzer::Visit_(const prim::LetNode* op) {
  if (auto result = this->Visit(op->value)) return result;
  analyzer_->Bind(op->var, op->value);
  return this->Visit(op->body);
}

PrimExpr IRVisitorWithAnalyzer::ExtractRealCondition(PrimExpr condition) const {
  if (auto call = condition.as<CallNode>()) {
    if (call->op.same_as(prim::builtin::likely())) {
      return call->args[0].as_or_throw<PrimExpr>();
    }
  }

  return condition;
}

}  // namespace tirx
}  // namespace tvm
