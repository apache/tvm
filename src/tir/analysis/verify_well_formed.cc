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
 * \file tir/analysis/verify_well_formed.cc
 * \brief Check if schedulable tir is well-formed.
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include "../ir/functor_common.h"

namespace tvm {
namespace tir {

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const PrimFunc& func, bool assert_mode) {
    BlockVarAccessVerifier verifier(assert_mode);
    verifier(func->body);
    return !verifier.has_error_;
  }

 private:
  explicit BlockVarAccessVerifier(bool assert_mode) : assert_mode_(assert_mode) {}

  void VisitStmt(const Stmt& stmt) final {
    if (!has_error_) {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }

  void VisitExpr(const PrimExpr& expr) final {
    if (!has_error_) {
      StmtExprVisitor::VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    auto it = loop_vars_.find(op);
    if (it != loop_vars_.end() && it->second < cur_block_level_) {
      has_error_ = true;
      if (assert_mode_) {
        report_error(op);
      }
    }
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(loop_vars_.find(op->loop_var.get()) == loop_vars_.end());
    loop_vars_[op->loop_var.get()] = cur_block_level_;
    StmtExprVisitor::VisitStmt_(op);
    loop_vars_.erase(op->loop_var.get());
  }

  void VisitStmt_(const BlockNode* op) final {
    // Do not check boundary if it's a opaque block.
    cur_block_level_ += !op->iter_vars.empty();

    // Step 0. Skip block iter var's domain

    // Step 1. Visit read/write regions
    auto fvisit_buffer_region = [this](const BufferRegion& s) {
      for (const auto& range : s->region) {
        this->VisitExpr(range->min);
        this->VisitExpr(range->extent);
      }
    };
    VisitArray(op->reads, fvisit_buffer_region);
    VisitArray(op->writes, fvisit_buffer_region);

    // Step 2. Visit match buffers
    VisitArray(op->match_buffers,
               [fvisit_buffer_region](const MatchBufferRegion& match_buffer_region) {
                 fvisit_buffer_region(match_buffer_region->source);
               });

    // Step 3. Visit init and body
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);

    cur_block_level_ -= !op->iter_vars.empty();
  }

 private:
  void report_error(const VarNode* var) {
    // TODO(siyuan): use the error message from the parser.
    LOG(FATAL) << "Well-formedness check failed: outside defined var " << var->name_hint
               << " is used inside the current block.";
  }

  /*! \brief The map from outside loop vars to its corresponding block level. */
  std::unordered_map<const VarNode*, size_t> loop_vars_;
  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Current nested block stack level. */
  size_t cur_block_level_{0};
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

bool VerifyWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!BlockVarAccessVerifier::Verify(func, assert_mode)) {
    return false;
  }
  // TODO(Siyuan): add more checks here.
  return true;
}

TVM_REGISTER_GLOBAL("tir.analysis.VerifyWellFormed").set_body_typed(VerifyWellFormed);

}  // namespace tir
}  // namespace tvm
