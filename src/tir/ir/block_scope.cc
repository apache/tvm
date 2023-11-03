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
#include <tvm/tir/block_scope.h>
#include <tvm/tir/utils.h>

namespace tvm {
namespace tir {

/******** Utility functions ********/

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Add a dependency relation.
 * \param src The source of the dependency
 * \param dst The destination of the dependecy
 * \param kind Type of the dependency
 * \note This method is effectively NOP on self-loops
 */
void AddDependency(BlockScopeNode* self, const StmtSRef& src, const StmtSRef& dst, DepKind kind) {
  if (!src.same_as(dst)) {
    Dependency dep(src, dst, kind);
    self->src2deps[src].push_back(dep);
    self->dst2deps[dst].push_back(dep);
  }
}

/******** Constructors ********/

StmtSRef::StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index) {
  ObjectPtr<StmtSRefNode> n = make_object<StmtSRefNode>();
  n->stmt = stmt;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

StmtSRef StmtSRef::InlineMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

StmtSRef StmtSRef::RootMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

Dependency::Dependency(StmtSRef src, StmtSRef dst, DepKind kind) {
  ObjectPtr<DependencyNode> node = make_object<DependencyNode>();
  node->src = std::move(src);
  node->dst = std::move(dst);
  node->kind = kind;
  data_ = std::move(node);
}

BlockScope::BlockScope() { data_ = make_object<BlockScopeNode>(); }

BlockScope::BlockScope(const Array<StmtSRef>& child_block_srefs) {
  ObjectPtr<BlockScopeNode> n = make_object<BlockScopeNode>();
  SMap<Buffer, Array<StmtSRef>> buffer_readers;
  SMap<Buffer, Array<StmtSRef>>& buffer_writers = n->buffer_writers;
  for (const StmtSRef& child_block_sref : child_block_srefs) {
    const BlockNode* child_block = TVM_SREF_TO_BLOCK(child_block_sref);
    // Step 1. Update `buffer_readers` and `buffer_writers` for each buffer
    for (const BufferRegion& region : child_block->reads) {
      buffer_readers[region->buffer].push_back(child_block_sref);
    }
    for (const BufferRegion& region : child_block->writes) {
      buffer_writers[region->buffer].push_back(child_block_sref);
    }
    // Step 2. Update RAW dependency
    for (const BufferRegion& region : child_block->reads) {
      auto it = buffer_writers.find(region->buffer);
      if (it != buffer_writers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kRAW);
        }
      }
    }
    // Step 3. Update WAW dependency
    for (const BufferRegion& region : child_block->writes) {
      auto it = buffer_writers.find(region->buffer);
      if (it != buffer_writers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kWAW);
        }
      }
    }
    // Step 4. Update WAR dependency
    for (const BufferRegion& region : child_block->writes) {
      auto it = buffer_readers.find(region->buffer);
      if (it != buffer_readers.end()) {
        for (const StmtSRef& from : it->second) {
          AddDependency(n.get(), from, child_block_sref, DepKind::kWAR);
        }
      }
    }
  }
  data_ = std::move(n);
}

/******** Dependency ********/

Array<Dependency> BlockScopeNode::GetDepsBySrc(const StmtSRef& block_sref) const {
  auto iter = this->src2deps.find(block_sref);
  if (iter != this->src2deps.end()) {
    return iter->second;
  } else {
    return {};
  }
}

Array<Dependency> BlockScopeNode::GetDepsByDst(const StmtSRef& block_sref) const {
  auto iter = this->dst2deps.find(block_sref);
  if (iter != this->dst2deps.end()) {
    return iter->second;
  } else {
    return {};
  }
}

/*!
 * \brief Add a new statement to the stack, which becomes the current scope
 * \param stmt A for-loop statement or a block statement
 */
void SRefTreeCreator::PushSRef(const StmtNode* stmt) {
  if (srefs_.empty()) {
    srefs_.push_back(
        StmtSRef(stmt,
                 /*parent=*/nullptr,
                 /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
  } else {
    StmtSRefNode* parent = srefs_.back().get();
    srefs_.push_back(
        StmtSRef(stmt, parent,
                 /*seq_index=*/-1));  // `seq_index` will be set properly in SetSeqIndex
  }
}

/*! \brief Pop the top of the scope and record it in stmt2ref map */
void SRefTreeCreator::PopAndRecordSRef() {
  StmtSRef sref = std::move(srefs_.back());
  stmt2ref_[sref->stmt] = sref;
  srefs_.pop_back();
}

void SRefTreeCreator::VisitStmt_(const ForNode* loop) {
  if (!include_loops_) {
    VisitStmt(loop->body);
  } else {
    PushSRef(loop);
    VisitStmt(loop->body);
    PopAndRecordSRef();
  }
}

void SRefTreeCreator::VisitStmt_(const BlockRealizeNode* realize) {
  const BlockNode* block = realize->block.get();
  PushSRef(block);
  VisitStmt(block->body);  // `block->init` is not visited
  PopAndRecordSRef();
}

void SRefTreeCreator::VisitStmt_(const SeqStmtNode* seq_stmt) {
  // Set `seq_index` information for SeqStmtNode
  StmtVisitor::VisitStmt_(seq_stmt);
  SetSeqIndexInChildren(stmt2ref_, seq_stmt, include_loops_);
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(StmtSRefNode);
TVM_REGISTER_NODE_TYPE(DependencyNode);
TVM_REGISTER_NODE_TYPE(BlockScopeNode);

TVM_REGISTER_GLOBAL("tir.StmtSRefStmt").set_body_typed([](StmtSRef sref) -> Optional<Stmt> {
  return GetRef<Optional<Stmt>>(sref->stmt);
});
TVM_REGISTER_GLOBAL("tir.StmtSRefParent").set_body_typed([](StmtSRef sref) -> Optional<StmtSRef> {
  return GetRef<Optional<StmtSRef>>(sref->parent);
});
TVM_REGISTER_GLOBAL("tir.StmtSRefRootMark")  //
    .set_body_typed(StmtSRef::RootMark);
TVM_REGISTER_GLOBAL("tir.StmtSRefInlineMark")  //
    .set_body_typed(StmtSRef::InlineMark);
TVM_REGISTER_GLOBAL("tir.BlockScopeGetDepsBySrc")
    .set_body_method<BlockScope>(&BlockScopeNode::GetDepsBySrc);
TVM_REGISTER_GLOBAL("tir.BlockScopeGetDepsByDst")
    .set_body_method<BlockScope>(&BlockScopeNode::GetDepsByDst);

}  // namespace tir
}  // namespace tvm
