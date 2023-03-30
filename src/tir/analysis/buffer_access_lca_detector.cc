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

#include "../../runtime/thread_storage_scope.h"
#include "../../support/arena.h"

namespace tvm {
namespace tir {

/*!
 * \brief Detect the lowest common ancestor(LCA) position of Buffer access.
 * \note
 * - Only consider BlockNode and ForNode to be the LCA nodes.
 * - In the LCA locator, we are aware of the buffer scope and CUDA hierarchy so that any buffer in
 * global memory will have its buffer access LCA outside all launch sites of `blockIdx`, in order to
 * prevent conflicts between buffer memory scopes and CUDA hierarchy.
 */
class LCADetector : public StmtExprVisitor {
 public:
  static Map<Buffer, Optional<Stmt>> Detect(const PrimFunc& func) {
    LCADetector detector;
    for (const auto& kv : func->buffer_map) {
      const Buffer& buffer = kv.second;
      detector.buffer_var_map_.emplace(buffer->data.get(), buffer.get());
    }

    // The root node must be explicitly present in the list of
    // ancestor_scopes_.  We cannot use nullptr to represent the root
    // node, as that is also used to represent a scope that hasn't
    // been observed before.
    ScopeInfo root(nullptr, nullptr, 0);
    detector.ancestor_scopes_.push_back(&root);

    detector(func->body);
    detector.UpdateWithBlockidx();

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

    if (op->thread_binding.defined()) {
      const runtime::ThreadScope& scope =
          runtime::ThreadScope::Create(op->thread_binding.value()->thread_tag);
      if (scope.rank == 0) {
        blockidx_scopes_.push_back(current_scope);
      }
    }

    ancestor_scopes_.push_back(current_scope);
    loop_scope_map_.insert({op->loop_var.get(), current_scope});
    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
    loop_scope_map_.erase(op->loop_var.get());
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    const BlockNode* block = op->block.get();
    int n = ancestor_scopes_.size();
    for (const Buffer& buf : block->alloc_buffers) {
      buffer_var_map_.emplace(buf->data.get(), buf.get());
    }

    const ScopeInfo* parent_scope = ancestor_scopes_.back();
    auto* current_scope = arena_.make<ScopeInfo>(parent_scope, block, n);

    ancestor_scopes_.push_back(current_scope);

    // For each accessed buffer of the block, update the buffer's lca to
    // the lowest inclusive stmt position, which should dominate all loops
    // related to the accessed opaque block iter vars in buffer indices.
    UpdateDominateScopeOfOpaqueIter(op);

    // Update match_buffers
    for (const MatchBufferRegion& match_buffer : block->match_buffers) {
      UpdateBufferLCA(match_buffer->source->buffer.get(), ancestor_scopes_.back());
      match_buffers_.insert(match_buffer->buffer.get());
    }

    StmtExprVisitor::VisitStmt_(op);
    ancestor_scopes_.pop_back();
  }

  void UpdateDominateScopeOfOpaqueIter(const BlockRealizeNode* block_realize) {
    // map opaque iter var to the scope which dominate all loop carried dependencies.
    std::unordered_map<const VarNode*, const ScopeInfo*> itervar_to_dom_scope;

    // function to collect `itervar_to_dom_scope`, the result scope for each block
    // iter var should be above all loop scopes the opaque iter var binding relates to.
    auto do_collect_itervar_scope = [this, &itervar_to_dom_scope](const IterVar& itervar,
                                                                  const PrimExpr& binding) {
      PostOrderVisit(binding, [this, &itervar_to_dom_scope, &itervar](const ObjectRef& obj) {
        if (const VarNode* loop_var = obj.as<VarNode>()) {
          auto it = loop_scope_map_.find(loop_var);
          if (it == loop_scope_map_.end()) {
            return;
          }
          const ScopeInfo* scope = it->second->parent_scope_info;
          // find the highest loop scope the iter var binding has related to.
          auto dom_scope_it = itervar_to_dom_scope.find(itervar->var.get());
          if (dom_scope_it == itervar_to_dom_scope.end()) {
            itervar_to_dom_scope.insert(dom_scope_it, {itervar->var.get(), scope});
          } else if (scope->depth < dom_scope_it->second->depth) {
            dom_scope_it->second = scope;
          }
        }
      });
    };

    // function to update lca scope of the buffer with loop carried dependent buffer accesses.
    // the result scope should be above all loop scopes the accessed opaque block iter vars
    // relate to, which is record in `itervar_to_dom_scope`.
    auto do_update = [this, &itervar_to_dom_scope](const BufferRegion& region) {
      const Buffer& buffer = region->buffer;
      const ScopeInfo* scope = ancestor_scopes_.back();

      auto handle_itervar = [&itervar_to_dom_scope, &scope](const ObjectRef& obj) {
        if (const VarNode* iter_var = obj.as<VarNode>()) {
          auto dom_scope_it = itervar_to_dom_scope.find(iter_var);
          if (dom_scope_it == itervar_to_dom_scope.end()) {
            return;
          }
          // find the highest loop scope the accessed buffer index has
          // loop carried dependencies to (via opaque iter var binding).
          if (dom_scope_it->second->depth < scope->depth) {
            scope = dom_scope_it->second;
          }
        }
      };

      // visit region min and max to find the lowest legal lca scope
      for (const Range& range : region->region) {
        PostOrderVisit(range->min, handle_itervar);
        PostOrderVisit(range->min + range->extent - 1, handle_itervar);
      }
      UpdateBufferLCA(buffer.get(), scope);
    };

    // do collect and update
    const Block& block = block_realize->block;
    for (size_t i = 0; i < block_realize->iter_values.size(); ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      if (iter_var->iter_type != IterVarType::kDataPar &&
          iter_var->iter_type != IterVarType::kCommReduce) {
        do_collect_itervar_scope(iter_var, block_realize->iter_values[i]);
      }
    }
    if (!itervar_to_dom_scope.empty()) {
      for (const auto& read : block->reads) {
        do_update(read);
      }
      for (const auto& write : block->writes) {
        do_update(write);
      }
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      const auto* iter = op->node.as<IterVarNode>();
      ICHECK_NOTNULL(iter);
      const runtime::ThreadScope& scope = runtime::ThreadScope::Create(iter->thread_tag);
      if (scope.rank == 0) {
        blockidx_scopes_.push_back(ancestor_scopes_.back());
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    UpdateBufferLCA(op->buffer.get(), ancestor_scopes_.back());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    UpdateBufferLCA(op->buffer.get(), ancestor_scopes_.back());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferRealizeNode* op) final {
    buffer_var_map_.emplace(op->buffer->data.get(), op->buffer.get());
    UpdateBufferLCA(op->buffer.get(), ancestor_scopes_.back());
    StmtExprVisitor::VisitStmt_(op);
  }

  // Works for Load/Store and opaque access.
  void VisitExpr_(const VarNode* op) final { VisitBufferVar(op); }

  void VisitBufferVar(const VarNode* op) {
    auto it = buffer_var_map_.find(op);
    if (it != buffer_var_map_.end()) {
      UpdateBufferLCA(it->second, ancestor_scopes_.back());
    }
  }

  void UpdateBufferLCA(const BufferNode* buffer, const ScopeInfo* scope) {
    buffer_var_map_.emplace(buffer->data.get(), buffer);
    if (match_buffers_.find(buffer) == match_buffers_.end()) {
      // Ingore buffer created by block match_buffer
      const ScopeInfo*& lca = buffer_lca_[buffer];
      lca = LowestCommonAncestor(lca, scope);
    }
  }

  void UpdateWithBlockidx() {
    for (const auto& it : buffer_lca_) {
      const runtime::StorageScope& scope =
          runtime::StorageScope::Create(GetRef<Buffer>(it.first).scope());
      if (scope.rank == runtime::StorageRank::kGlobal) {
        const ScopeInfo*& lca = buffer_lca_[it.first];
        for (const ScopeInfo* blockidx_scope : blockidx_scopes_) {
          lca = LowestCommonAncestor(lca, blockidx_scope);
        }
      }
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

  /*! \brief The ancestor scope stacks info (Block and For).  The
   *  first element is initialized in LCADetector::Detect to represent
   *  the root scope.
   */
  std::vector<const ScopeInfo*> ancestor_scopes_ = {};
  /*! \brief The map from Buffer to its LCA ForNode/BlockNode. */
  std::unordered_map<const BufferNode*, const ScopeInfo*> buffer_lca_ = {};
  /*! \brief The map from Buffer data to the Buffer. */
  std::unordered_map<const VarNode*, const BufferNode*> buffer_var_map_ = {};
  /*! \brief The match buffers inside blocks. */
  std::unordered_set<const BufferNode*> match_buffers_ = {};
  /*! \brief The ForNodes/BlockNodes which contain immediate `blockIdx` launch. */
  std::vector<const ScopeInfo*> blockidx_scopes_ = {};
  /*! \brief The map from loop var to the corresponding scope. */
  std::unordered_map<const VarNode*, const ScopeInfo*> loop_scope_map_ = {};
  /*! \brief Internal arena. */
  support::Arena arena_;
};

Map<Buffer, Optional<Stmt>> DetectBufferAccessLCA(const PrimFunc& func) {
  return LCADetector::Detect(func);
}

TVM_REGISTER_GLOBAL("tir.analysis.detect_buffer_access_lca").set_body_typed(DetectBufferAccessLCA);
}  // namespace tir
}  // namespace tvm
