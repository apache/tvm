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
 * \file tir/analysis/buffer_access_lca_detector.cc
 * \brief Detect the lowest common ancestor(LCA) of buffer access
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "../../support/arena.h"

namespace tvm {
namespace tir {

/*!
 * \brief Detect the lowest common ancestor(LCA) position of Buffer access.
 * \note Only consider BlockNode and ForNode to be the LCA nodes.
 */
class LCADetector : public StmtExprVisitor {
 public:
  static Map<Buffer, Optional<Stmt>> Detect(const PrimFunc& func) {
    LCADetector detector;
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      detector.buffer_var_map_.emplace(buffer->data.get(), buffer.get());
    }

    detector(func->body);
    // Prepare the return
    Map<Buffer, Optional<Stmt>> buffer_lca;
    for (const auto& kv : detector.buffer_lca_) {
      const Buffer& buffer = GetRef<Buffer>(kv.first);
      const Optional<Stmt> stmt = kv.second ? GetRef<Optional<Stmt>>(kv.second->stmt) : NullOpt;
      buffer_lca.Set(buffer, stmt);
    }
    return buffer_lca;
  }

 private:
  /*!
   * \brief The AST node information for querying LCA.
   * \note Only BlockNode and ForNode are considered, since they are the only statements whose
   *       body can be a SeqStmt (the LCA of buffer access) in TensorIR.
   */
  struct ScopeInfo {
    // The parent scope info
    const ScopeInfo* parent_scope_info;
    // The parent scope stmt node
    const StmtNode* stmt;
    // The scope depth in the AST
    int depth;
    ScopeInfo(const ScopeInfo* parent_info, const StmtNode* stmt, int depth)
        : parent_scope_info(parent_info), stmt(stmt), depth(depth) {}
  };

  void VisitStmt_(const ForNode* op) final {
    int n = ancestor_scopes_.size();
    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, op, n);
    ancestor_scopes_.push_back(current_scope);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    int n = ancestor_scopes_.size();
    for (const Buffer& buf : op->alloc_buffers) {
      buffer_var_map_.emplace(buf->data.get(), buf.get());
    }

    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, op, n);

    ancestor_scopes_.push_back(current_scope);
    // Update match_buffers
    for (const MatchBufferRegion& match_buffer : op->match_buffers) {
      UpdateBufferLCA(match_buffer->source->buffer.get());
      match_buffers_.insert(match_buffer->buffer.get());
    }

    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    UpdateBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    UpdateBufferLCA(op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode* op) final {
    buffer_var_map_.emplace(op->buffer->data.get(), op->buffer.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  // Works for Load/Store and opaque access.
  void VisitExpr_(const VarNode* op) final { VisitBufferVar(op); }

  // Explict to visit buffer data in Load and Store node.
  void VisitExpr_(const LoadNode* op) final {
    ExprVisitor::VisitExpr_(op);
    VisitBufferVar(op->buffer_var.get());
  }

  void VisitStmt_(const StoreNode* op) final {
    StmtVisitor::VisitStmt_(op);
    VisitBufferVar(op->buffer_var.get());
  }

  void VisitBufferVar(const VarNode* op) {
    auto it = buffer_var_map_.find(op);
    if (it != buffer_var_map_.end()) {
      UpdateBufferLCA(it->second);
    }
  }

  void UpdateBufferLCA(const BufferNode* buffer) {
    if (match_buffers_.find(buffer) == match_buffers_.end()) {
      // Ingore buffer created by block match_buffer
      const ScopeInfo*& lca = buffer_lca_[buffer];
      lca = LowestCommonAncestor(lca, ancestor_scopes_.back());
    }
  }

  static const ScopeInfo* LowestCommonAncestor(const ScopeInfo* lhs, const ScopeInfo* rhs) {
    if (lhs == nullptr) return rhs;
    if (rhs == nullptr) return lhs;
    while (lhs->parent_scope_info != nullptr &&  //
           rhs->parent_scope_info != nullptr &&  //
           lhs != rhs) {
      if (lhs->depth == rhs->depth) {
        lhs = lhs->parent_scope_info;
        rhs = rhs->parent_scope_info;
      } else if (lhs->depth < rhs->depth) {
        rhs = rhs->parent_scope_info;
      } else {
        lhs = lhs->parent_scope_info;
      }
    }
    if (lhs->parent_scope_info == nullptr) {
      return lhs;
    }
    if (rhs->parent_scope_info == nullptr) {
      return rhs;
    }
    ICHECK(lhs == rhs);
    return lhs;
  }

  /*! \brief The ancestor scope stacks info (Block and For), initialized with Null. */
  std::vector<const ScopeInfo*> ancestor_scopes_ = {nullptr};
  /*! \brief The map from Buffer to its LCA ForNode/BlockNode. */
  std::unordered_map<const BufferNode*, const ScopeInfo*> buffer_lca_ = {};
  /*! \brief The map from Buffer data to the Buffer. */
  std::unordered_map<const VarNode*, const BufferNode*> buffer_var_map_ = {};
  /*! \brief The match buffers inside blocks. */
  std::unordered_set<const BufferNode*> match_buffers_ = {};
  /*! \brief Internal arena. */
  support::Arena arena_;
};

Map<Buffer, Optional<Stmt>> DetectBufferAccessLCA(const PrimFunc& func) {
  return LCADetector::Detect(func);
}

TVM_REGISTER_GLOBAL("tir.analysis.detect_buffer_access_lca").set_body_typed(DetectBufferAccessLCA);
}  // namespace tir
}  // namespace tvm
