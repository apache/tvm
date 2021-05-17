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
#include "../utils.h"

namespace tvm {
namespace tir {

class SRefTreeVerifier : public StmtVisitor {
 public:
  static void Verify(const ScheduleStateNode* self) { SRefTreeVerifier(self).Verify(); }

 private:
  /*! \brief Constructor */
  explicit SRefTreeVerifier(const ScheduleStateNode* self) : self_(self) {}

  void Verify() {
    VisitPrimFuncs(self_->mod, [this](const PrimFuncNode* func) { this->VisitStmt(func->body); });
    ICHECK_EQ(n_sref_visited_, static_cast<int>(self_->stmt2ref.size()));
    for (const auto& kv : self_->block_info) {
      const StmtSRef& sref = kv.first;
      ICHECK(sref->stmt != nullptr)
          << "InternalError: An expired sref is found in the block_scope mapping";
      auto it = self_->stmt2ref.find(sref->stmt);
      ICHECK(it != self_->stmt2ref.end())
          << "InternalError: The sref points to a statement that does not exist in stmt2ref";
      const StmtSRef& sref2 = it->second;
      ICHECK(sref.same_as(sref2))
          << "InternalError: The sref points to a statement whose corresponding sref in stmt2ref "
             "is not the same object as itself";
    }
    ICHECK_EQ(n_block_sref_visited_, static_cast<int>(self_->block_info.size()));
  }

  void VisitStmt_(const BlockNode* block) final {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(block)) << "InternalError: A block inside init block has its "
                                               "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(block);
      return;
    }
    ICHECK(self_->stmt2ref.count(block))
        << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(block);
    ++n_sref_visited_;
    ++n_block_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(block);
    ICHECK(self_->block_info.count(sref))
        << "InternalError: Cannot find scope information of the BlockNode:\n"
        << GetRef<Stmt>(block);
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for BlockNode:\n"
        << GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    if (block->init.defined()) {
      ++init_block_depth_;
      VisitStmt(block->init.value());
      --init_block_depth_;
    }
    VisitStmt(block->body);
    ancestors_.pop_back();
  }

  void VisitStmt_(const ForNode* loop) final {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(loop)) << "InternalError: A loop inside init block has its "
                                              "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(loop);
      return;
    }
    ICHECK(self_->stmt2ref.count(loop))
        << "InternalError: A ForNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(loop);
    ++n_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(loop);
    Optional<Stmt> stmt = NullOpt;
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for ForNode:\n"
        << GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(loop);
    ancestors_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Verify seq_index
    if (init_block_depth_) {
      StmtVisitor::VisitStmt_(seq_stmt);
      return;
    }
    int n = static_cast<int>(seq_stmt->seq.size());
    for (int i = 0; i < n; ++i) {
      const Stmt& child = seq_stmt->seq[i];
      StmtSRef sref{nullptr};
      if (const auto* realize = child.as<BlockRealizeNode>()) {
        const auto* block = realize->block.get();
        ICHECK(self_->stmt2ref.count(block));
        sref = self_->stmt2ref.at(block);
      } else if (child->IsInstance<ForNode>()) {
        ICHECK(self_->stmt2ref.count(child.get()));
        sref = self_->stmt2ref.at(child.get());
      } else {
        continue;
      }
      ICHECK_EQ(sref->seq_index, i) << "InternalError: A StmtSRef has incorrect seq_index";
    }
    StmtVisitor::VisitStmt_(seq_stmt);
  }

  /*! \brief The schedule it belongs to */
  const ScheduleStateNode* self_;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefNode*> ancestors_ = {nullptr};
  /*! \brief If the visitor is currently in the init block */
  int init_block_depth_ = 0;
  /*! \brief Number of srefs that are visited */
  int n_sref_visited_ = 0;
  /*! \brief Number of block srefs that are visited */
  int n_block_sref_visited_ = 0;
};

void VerifySRefTree(const ScheduleState& self) { SRefTreeVerifier::Verify(self.get()); }

}  // namespace tir
}  // namespace tvm
