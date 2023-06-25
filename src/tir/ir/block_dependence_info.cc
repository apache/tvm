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

#include <tvm/tir/block_dependence_info.h>
#include <tvm/tir/utils.h>

namespace tvm {
namespace tir {

/**
 * @brief A helper class to collect and build Block Dependences using BlockScope class
 */
class BlockDependenceInfoCollector : private StmtVisitor {
 public:
  static void Collect(BlockDependenceInfoNode* self, const Stmt& stmt) {
    BlockDependenceInfoCollector collector(self);
    collector.VisitStmt(stmt);
  }

  explicit BlockDependenceInfoCollector(BlockDependenceInfoNode* self)
      : self_(self), block_frames_{} {
    block_frames_.emplace_back();
  }

  void MakeBlockScope(StmtSRef scope) {
    Array<StmtSRef> child_block_srefs = std::move(block_frames_.back());
    self_->sref2scope[scope] = BlockScope(child_block_srefs);
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    block_frames_.emplace_back();
    const BlockNode* block = realize->block.get();
    // Recursive visit
    VisitStmt(block->body);  // `block->init` is not visited
    // Create BlockInfo for the block
    auto sref = self_->stmt2ref.at(block);
    MakeBlockScope(sref);
    // Update parent scope
    block_frames_.pop_back();
    block_frames_.back().push_back(sref);
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Set `seq_index` information for SeqStmtNode
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndexInChildren(self_->stmt2ref, seq_stmt, false);
  }

  BlockDependenceInfoNode* self_;
  /*! \brief The stack frames of blocks in the DFS visit. */
  std::vector<Array<StmtSRef>> block_frames_;
};

BlockDependenceInfo::BlockDependenceInfo() { data_ = make_object<BlockDependenceInfoNode>(); }

BlockDependenceInfo::BlockDependenceInfo(IRModule mod) {
  ObjectPtr<BlockDependenceInfoNode> n = make_object<BlockDependenceInfoNode>();
  BlockDependenceInfoNode* self = n.get();
  n->stmt2ref = SRefTreeCreator::Create(mod, /* include_loops */ false);

  for (const auto& kv : mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (auto opt = base_func.as<PrimFunc>()) {
      auto func = opt.value();
      BlockDependenceInfoCollector::Collect(self, func->body);
    }
  }
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(BlockDependenceInfoNode);
TVM_REGISTER_GLOBAL("tir.BlockDependenceInfo")
    .set_body_typed([](IRModule mod) -> BlockDependenceInfo { return BlockDependenceInfo(mod); });
TVM_REGISTER_GLOBAL("tir.BlockDependenceInfoGetBlockScope")
    .set_body_method<BlockDependenceInfo>(&BlockDependenceInfoNode::GetBlockScope);
TVM_REGISTER_GLOBAL("tir.BlockDependenceInfoGetSRef")
    .set_body_typed([](BlockDependenceInfo self, Stmt stmt) -> Optional<StmtSRef> {
      auto it = self->stmt2ref.find(stmt.get());
      return it != self->stmt2ref.end() ? it->second : Optional<StmtSRef>(NullOpt);
    });

}  // namespace tir
}  // namespace tvm
