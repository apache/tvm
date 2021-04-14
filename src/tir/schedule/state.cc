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
#include "./utils.h"

namespace tvm {
namespace tir {

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

/**************** Utility functions ****************/

/*!
 * \brief Set the `StmtSRefNode::seq_index` field for stmt
 * \param self The schedule class
 * \param stmt The statement, or the realize node of the statement whose sref to be set
 * \param seq_index The seq_index to be set
 * \note The method is NOP for statements that are not scheduleable, i.e. not For or Block
 */
void SetSeqIndex(ScheduleStateNode* self, const Stmt& stmt, int seq_index) {
  if (const auto* realize = stmt.as<BlockRealizeNode>()) {
    const BlockNode* block = realize->block.get();
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* block = stmt.as<BlockNode>()) {
    ICHECK(self->stmt2ref.count(block));
    self->stmt2ref.at(block)->seq_index = seq_index;
  } else if (const auto* loop = stmt.as<ForNode>()) {
    ICHECK(self->stmt2ref.count(loop));
    self->stmt2ref.at(loop)->seq_index = seq_index;
  } else {
    // do nothing
  }
}

/*!
 * \brief Update seq_index of the children of a SeqStmt
 * \param self The schedule class
 * \param seq_stmt The SeqStmt whose children need updating
 */
void SetSeqIndexInChildren(ScheduleStateNode* self, const SeqStmtNode* seq_stmt) {
  int i = 0;
  for (const Stmt& stmt : seq_stmt->seq) {
    SetSeqIndex(self, stmt, i);
    ++i;
  }
}

/*!
 * \brief Update the sref information on the schedule class, as well as the statement of sref itself
 * More specifically, update
 *  `sref->stmt` to `new_stmt`
 *  `self->stmt2ref`, remove the old statement that sref points to, and add the new statement
 * \param self The schedule class to be updated
 * \param sref The sref to be updated
 * \param new_stmt The statement that replaces the statement inside the sref
 */
void UpdateSRef(ScheduleStateNode* self, StmtSRefNode* sref, const StmtNode* new_stmt) {
  ICHECK(new_stmt->IsInstance<BlockNode>() || new_stmt->IsInstance<ForNode>());
  const StmtNode* old_stmt = sref->stmt;
  ICHECK_NE(new_stmt, old_stmt);
  self->stmt2ref[new_stmt] = GetRef<StmtSRef>(sref);
  self->stmt2ref.erase(sref->stmt);
  sref->stmt = new_stmt;
}

/*!
 * \brief Get PrimFunc and GlobalVar that the root block belongs to
 * \param mod The IRModule
 * \param root_block The root block of the PrimFunc
 * \param result_g_var The result GlobalVar
 * \return The result PrimFunc where the root block belongs to
 * \note This function returns the pointer instead of ObjectRef to avoid later copy-on-write
 */
const PrimFuncNode* GetRootPrimFunc(const IRModule& mod, const StmtNode* root_block,
                                    GlobalVar* result_g_var) {
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == root_block) {
          *result_g_var = g_var;
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the correpsonding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(root_block);
  throw;
}

/**************** Creation ****************/

/*! \brief A helper class to create a new ScheduleStateNode from an IRModule */
class StateCreator : private StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule state to be completed
   */
  static ObjectPtr<ScheduleStateNode> Create(IRModule mod, int debug_mode) {
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    ScheduleStateNode* self = n.get();
    // Set `n->mod`
    n->mod = std::move(mod);
    // Set `n->debug_mode`
    n->debug_mode = debug_mode;
    // Set `n->stmt2ref` and `n->block_info`
    StateCreator creator(self);
    for (const auto& kv : n->mod->functions) {
      const BaseFunc& base_func = kv.second;
      if (const auto* func = base_func.as<PrimFuncNode>()) {
        creator.VisitStmt(func->body);
      }
    }
    return n;
  }

 private:
  explicit StateCreator(ScheduleStateNode* self)
      : self_(self), srefs_{}, realizes_{}, block_frames_{} {
    block_frames_.emplace({});
  }

  /*!
   * \brief Add a new statement to the stack, which becomes the current scope
   * \param stmt A for-loop statement or a block statement
   * \return A sref to the stmt
   */
  StmtSRef PushSRef(const StmtNode* stmt) {
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
    return srefs_.back();
  }

  /*! \brief Pop the top of the scope and record it in stmt2ref map */
  StmtSRef PopAndRecordSRef() {
    StmtSRef sref = std::move(srefs_.back());
    self_->stmt2ref[sref->stmt] = sref;
    srefs_.pop_back();
    return sref;
  }

  void MakeBlockInfo(StmtSRef scope_root) {
    // Calculate `BlockInfo::scope`
    Array<StmtSRef> child_block_srefs = std::move(block_frames_.back());
    BlockInfo& info =
        self_->block_info.emplace(std::move(scope_root), BlockInfo(BlockScope(child_block_srefs)))
            .first->second;
    // TODO(@junrushao1994): calculate the flags
    // Set `affine_binding`
    info.affine_binding = false;
    // Set `region_cover`
    info.region_cover = false;
    // Set `stage_pipeline`
    info.scope->stage_pipeline = false;
  }

  void VisitStmt_(const ForNode* loop) final {
    PushSRef(loop);
    VisitStmt(loop->body);
    PopAndRecordSRef();
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    realizes_.push_back(realize);
    block_frames_.emplace_back();
    const BlockNode* block = realize->block.get();
    // Recursive visit
    PushSRef(block);
    VisitStmt(block->body);  // `block->init` is not visited
    StmtSRef sref = PopAndRecordSRef();
    // Create BlockInfo for the block
    MakeBlockInfo(sref);
    // Update parent scope
    block_frames_.pop_back();
    block_frames_.back().push_back(sref);
    realizes_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    // Set `seq_index` information for SeqStmtNode
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndexInChildren(self_, seq_stmt);
  }

  /*! \brief The result ScheduleStateNode */
  ScheduleStateNode* self_;
  /*! \brief The stack frame used to indicate the current scope */
  std::vector<StmtSRef> srefs_;
  /*! \brief The BlockRealize in the ancestors */
  std::vector<const BlockRealizeNode*> realizes_;
  /*! \brief The stack frames of blocks in the DFS visit. */
  std::vector<Array<StmtSRef>> block_frames_;
};

/**************** Constructor ****************/

ScheduleState::ScheduleState(IRModule mod, int debug_mode) {
  CHECK_GE(debug_mode, -1) << "ValueError: negative `debug_mode` other than -1 is not supported";
  data_ = StateCreator::Create(mod, debug_mode);
  (*this)->DebugVerify();
}

ScheduleState::ScheduleState(PrimFunc func, int debug_mode)
    : ScheduleState(IRModule({{GlobalVar("main"), func}}), debug_mode) {}

/**************** Replace ****************/

/*
 * The goal of the replacement algorithm is to substitute a subtree `src_stmt` of the AST to a new
 * subtree `tgt_stmt`, and maintain the corresponding sref tree accordingly, with some srefs reused,
 * so that the srefs users hold doesn't expire. For example, if we split a loop into 2, and the
 * original loop has a child block, then the sref to the child block should be reused, so that users
 * won't have to acquire that sref again.
 *
 * The workflow of the replacement algorithm is:
 * 1) Detect all possible reuses in class ReuseInfo
 * 2) Remove the expired srefs in class SRefTreePruner
 * 3) Update the reused the sref, and create the srefs for new statements, in class SRefUpdater
 * 4) Renew the ancestors of `src_stmt` to reflect the replacement
 */

/*!
 * \brief Record the different sref reuse types in the replacement
 *
 * 1) Intact: the subtree appears as the same object on both `src_stmt` and `tgt_stmt`,
 * which, given the immutability of the IR, means the entire subtree is unchanged,
 * and we do not need to recurse into the subtree.
 *
 * 2) Loop/Block sref reuse: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and change it to point to `tgt`.
 *
 * \note The intact reuse and loop sref reuse are collected in the ReuseCollector,
 * while the block reuse is specified by the caller.
 *
 * \sa ReuseCollector
 */
struct ReuseInfo {
  /*!
   * \brief Kind 1. Intact reuse. If a stmt is in `intact`, it means its corresponding
   * sref is reused and it is intact reuse.
   */
  std::unordered_set<const StmtNode*> intact;
  /*!
   * \brief Kind 2.1. Loop sref reuse
   * If the loop var of a loop is in `loop_sref_possible_reuse`,
   * it means that when `src_stmt` has a loop that uses this loop var,
   * the reuse kind is loop sref reuse.
   * \note For each loop var in `loop_sref_possible_reuse`, it is possible that `src_stmt` doesn't
   * contain a loop that uses this loop var, and that is the reason why it is named "possible".
   */
  std::unordered_set<const VarNode*> loop_sref_possible_reuse;
  /*!
   * \brief Kind 2.2. Block sref reuse.
   * Maps an old Block in `src_stmt` to a new block in `tgt_stmt`,
   * indicating the sref to the old block should be reused in the sref to the new block.
   */
  std::unordered_map<const BlockNode*, const BlockNode*> block_sref_reuse;
};

/*!
 * \brief A helper visitor which collects two cases of sref reuses in the `tgt_stmt`:
 *
 * 1) Intact: the subtree represented by `intact` appears on both old and new IR.
 * Given the immutability of the IR, we can quickly decide that the entire subtree is unchanged,
 * which means we do not need to visit into the subtree of the old statement.
 *
 * 2) Reused block/loop: for two different objects (`src`, `tgt`),
 * which are both loops or both blocks,
 * and there is correspondence between them,
 * which makes us to reuse the sref pointing to `src`, and changes it to point to `tgt`,
 */
class ReuseCollector : public StmtVisitor {
 public:
  static ReuseInfo Collect(const ScheduleStateNode* self, const Stmt& tgt_stmt) {
    ReuseCollector collector(self);
    collector.VisitStmt(tgt_stmt);
    ReuseInfo result;
    result.intact = {collector.intact_.begin(), collector.intact_.end()};
    result.loop_sref_possible_reuse = {collector.loop_vars_.begin(), collector.loop_vars_.end()};
    // `result.block_reuse ` is not set here because ReuseCollector doesn't collect it,
    // and it is supposed to be properly set by the caller.
    return result;
  }

 private:
  explicit ReuseCollector(const ScheduleStateNode* self) : self_(self) {}

  void VisitStmt_(const ForNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      // Collect loop vars for detecting reuse of loop sref
      loop_vars_.push_back(op->loop_var.get());
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    if (self_->stmt2ref.count(op)) {
      intact_.push_back(op);
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }

  /*! \brief The schedule state to be worked on */
  const ScheduleStateNode* self_;
  /*! \brief The intact statements we have collected along the way of visiting */
  std::vector<const StmtNode*> intact_;
  /*! \brief The loop variable we collected in the tgt_stmt */
  std::vector<const VarNode*> loop_vars_;
};

/*!
 * \brief A helper visitor which removes the stale srefs in the `src_stmt`
 * that are useless after the replacement.
 *
 * It uses the reuse information previously collected to
 * 1) delete those srefs that are not reused.
 * 2) return the sref objects that are loop/block sref reuses, but not intact reuses
 */
class SRefTreePruner : public StmtVisitor {
 public:
  /*!
   * \brief The entry function
   * \param self The schedule class
   * \param info The reuse info about intact reuse and loop/block reuse
   * \param src_stmt The `src_stmt` where stale srefs to be removed
   * \return Mapping from the reuse elements to reused srefs, more specifically:
   * 1) Loop reuse: maps a loop var to the reused sref
   * 2) Block reuse: maps a block stmt to the reused sref,
   * where the block comes from the subtree of `tgt_stmt`
   * 3) Intact reuse: not returned
   */
  static std::unordered_map<const Object*, StmtSRef> Prune(ScheduleStateNode* self,
                                                           const ReuseInfo& reuse_info,
                                                           const Stmt& src_stmt) {
    SRefTreePruner pruner(self, reuse_info);
    pruner.VisitStmt(src_stmt);
    return std::move(pruner.reused_srefs_);
  }

 private:
  explicit SRefTreePruner(ScheduleStateNode* self, const ReuseInfo& reuse_info)
      : self_(self), reuse_info_(reuse_info) {}

  void VisitStmt_(const ForNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the loop:\n"
        << GetRef<For>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    const VarNode* loop_var = op->loop_var.get();
    if (reuse_info_.loop_sref_possible_reuse.count(loop_var)) {
      // sref can be reused
      reused_srefs_.emplace(loop_var, std::move(sref));
    } else {
      sref->Reset();
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    VisitStmt(op->body);
  }

  void VisitStmt_(const BlockNode* op) final {
    if (reuse_info_.intact.count(op)) {
      return;
    }
    auto it = self_->stmt2ref.find(op);
    ICHECK(it != self_->stmt2ref.end())
        << "IndexError: Cannot find correpsonding StmtSRef for the block:\n"
        << GetRef<Block>(op);
    StmtSRef& sref = it->second;
    // Detect reuse
    auto reuse_it = reuse_info_.block_sref_reuse.find(op);
    if (reuse_it != reuse_info_.block_sref_reuse.end()) {
      // sref can be reused
      reused_srefs_.emplace(reuse_it->second, std::move(sref));
    } else {
      sref->Reset();
      self_->block_info.erase(sref);
    }
    // erase the statement
    self_->stmt2ref.erase(it);
    // detect recursively
    // op->init is omitted
    VisitStmt(op->body);
  }

  /*! \brief The schedule state we are working on */
  ScheduleStateNode* self_;
  /*! \brief The reuse information we collected previously */
  const ReuseInfo& reuse_info_;
  /*!
   * \brief Reused srefs:
   * 1) loop var -> StmtSRef
   * 2) block stmt -> StmtSRef, where the block comes from the subtree of `tgt_stmt`
   */
  std::unordered_map<const Object*, StmtSRef> reused_srefs_;
};

/*!
 * \brief Update the sref in the `tgt_stmt` given the reuse information
 *
 * After being updated, in the `tgt_stmt` subtree,
 * 1) all `StmtSRefNode::parent`s are correct
 * 2) all `StmtSRefNode::seq_index`s are correct, except for the root
 * 3) all `StmtSRefNode::stmt`s are correct, except for the root
 */
class SRefUpdater : public StmtVisitor {
 public:
  static void Update(ScheduleStateNode* self, StmtSRefNode* src_stmt_parent,
                     const std::unordered_map<const Object*, StmtSRef>& reused_srefs,
                     const Stmt& tgt_stmt) {
    SRefUpdater(self, src_stmt_parent, reused_srefs).VisitStmt(tgt_stmt);
  }

 private:
  explicit SRefUpdater(ScheduleStateNode* self, StmtSRefNode* src_stmt_parent,
                       const std::unordered_map<const Object*, StmtSRef>& reused_srefs)
      : self_(GetRef<ScheduleState>(self)),
        ancestors_{src_stmt_parent},
        reused_srefs_(reused_srefs) {}

  void VisitStmt_(const ForNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact reuse
    if (sref.defined()) {
      sref->parent = ancestors_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
      return;
    }
    // Detect loop reuse
    auto it = reused_srefs_.find(op->loop_var.get());
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op->loop_var]`
      sref = it->second;
      sref->stmt = op;
      sref->parent = ancestors_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
    } else {
      // A new loop sref without reuse
      sref = StmtSRef(/*stmt=*/op, /*parent=*/ancestors_.back(),
                      /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    }
    // Recursive visit
    ancestors_.push_back(sref.get());
    VisitStmt(op->body);
    ancestors_.pop_back();
  }

  void VisitStmt_(const BlockNode* op) final {
    StmtSRef& sref = self_->stmt2ref[op];
    // Detect intact
    if (sref.defined()) {
      sref->parent = ancestors_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
      return;
    }
    // Detect block reuse
    auto it = reused_srefs_.find(op);
    if (it != reused_srefs_.end()) {
      // Update `stmt2ref[op]` to `reused_srefs_[op]`
      sref = it->second;
      sref->stmt = op;
      sref->parent = ancestors_.back();
      sref->seq_index = -1;  // `seq_index` will be set properly in SetSeqIndex
    } else {
      // A new block sref without reuse
      sref = StmtSRef(/*stmt=*/op, /*parent=*/ancestors_.back(),
                      /*seq_index=*/-1);  // `seq_index` will be set properly in SetSeqIndex
    }
    // Recursive visit
    ancestors_.push_back(sref.get());
    VisitStmt(op->body);
    ancestors_.pop_back();
    // Additionally, need to update the scope because the block is changed
    UpdateBlockInfo(sref);
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    StmtVisitor::VisitStmt_(seq_stmt);
    SetSeqIndexInChildren(self_.get(), seq_stmt);
  }

  void UpdateBlockInfo(const StmtSRef& block_sref) {
    using TIter = std::unordered_map<StmtSRef, BlockInfo, ObjectPtrHash, ObjectPtrEqual>::iterator;
    // The caller is responsible for correcting the flags
    BlockInfo new_info(BlockScope(GetChildBlocks(self_, block_sref)));
    std::pair<TIter, bool> insert_result = self_->block_info.emplace(block_sref, new_info);
    bool inserted = insert_result.second;
    BlockInfo& info = insert_result.first->second;
    if (inserted) {
      // Insertion has happened, update the flags accordingly
      BlockInfo& info = insert_result.first->second;
      info.affine_binding = false;
      info.region_cover = false;
      info.scope->stage_pipeline = false;
    } else {
      // Insertion didn't take place, because the entry has been there before.
      // In this case, we assume that flags are still valid so intentionally keep them unchanged
      info.scope = std::move(new_info.scope);
    }
  }

  /*! \brief The schedule state class to be worked on */
  ScheduleState self_;
  /*! \brief A stack containing all the ancestor For/Block nodes during the visit */
  std::vector<StmtSRefNode*> ancestors_;
  /*! \brief Maps the loop var / block to the reused sref */
  const std::unordered_map<const Object*, StmtSRef>& reused_srefs_;
};

/*!
 * \brief A helper that returns a new copy of `parent_stmt`,
 * where the subtree `child_src_stmt` is replaced with the subtree `child_tgt_stmt`.
 * \note The visitor assumes `child_src_stmt` is the child of `parent_stmt` in the sref tree.
 */
class ChildReplacer : private StmtMutator {
 public:
  static Stmt Replace(const StmtNode* parent_stmt, const StmtNode* child_src_stmt,
                      const Stmt& child_tgt_stmt, int seq_index, bool allow_copy_on_write) {
    // Check the invariant
    ICHECK(child_src_stmt->IsInstance<BlockNode>() ||  //
           child_src_stmt->IsInstance<ForNode>());
    ICHECK(child_tgt_stmt->IsInstance<BlockNode>() ||  //
           child_tgt_stmt->IsInstance<ForNode>() ||    //
           child_tgt_stmt->IsInstance<BlockRealizeNode>());
    ChildReplacer replacer(child_src_stmt, child_tgt_stmt, seq_index);
    replacer.allow_copy_on_write_ = allow_copy_on_write;
    return replacer.CopyOnWriteAndVisit(parent_stmt);
  }

 private:
  explicit ChildReplacer(const StmtNode* src_stmt, const Stmt& tgt_stmt, int seq_index)
      : src_stmt_(src_stmt), tgt_stmt_(tgt_stmt), seq_index_(seq_index) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    if (stmt.get() == src_stmt_) {
      // If the statement matches the `src_stmt` to be replaced, just return the `tgt_stmt`
      return tgt_stmt_;
    } else {
      return StmtMutator::VisitStmt(stmt);
    }
  }

  // Skipping sibling blocks and loops other than `src_stmt_`
  Stmt VisitStmt_(const BlockNode* op) final { return GetRef<Stmt>(op); }
  Stmt VisitStmt_(const ForNode* op) final { return GetRef<Stmt>(op); }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    int i = this->seq_index_;
    int n = static_cast<int>(op->seq.size());
    if (0 <= i && i < n) {
      const Stmt& stmt = op->seq[i];
      Optional<Stmt> new_stmt = NullOpt;
      const StmtNode* src_stmt = this->src_stmt_;
      // `stmt` can be For or BlockRealize
      // `src_stmt` can be For or Block
      // so the match from `stmt` to `src_stmt` can be
      // 1) For -> For
      // 2) BlockRealize -> Block
      if (stmt.get() == src_stmt) {
        // Case 1. src_stmt is For, stmt is For
        new_stmt = tgt_stmt_;
      } else if (const auto* realize = stmt.as<BlockRealizeNode>()) {
        // Case 2. stmt is BlockRealize, src_stmt is Block
        if (realize->block.get() == src_stmt) {
          const auto* tgt_block = TVM_TYPE_AS(tgt_block, tgt_stmt_, BlockNode);
          ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
          new_realize->block = GetRef<Block>(tgt_block);
          new_stmt = BlockRealize(std::move(new_realize));
        }
      }
      // Move new_stmt to position i
      if (new_stmt.defined()) {
        ObjectPtr<SeqStmtNode> new_seq_stmt = CopyOnWrite(op);
        new_seq_stmt->seq.Set(i, new_stmt.value());
        return SeqStmt(std::move(new_seq_stmt));
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt CopyOnWriteAndVisit(const StmtNode* parent_stmt) {
    // Step 1. Copy-on-write the `parent_stmt` and extract its `body`,
    // where `body` means the body of either a block or a loop
    // Step 2. Mutate the `block/loop->body`, searching for `child_old_stmt`
    // and replace it with `child_tgt_stmt`
    if (parent_stmt->IsInstance<BlockNode>()) {
      auto* block = const_cast<BlockNode*>(static_cast<const BlockNode*>(parent_stmt));
      ObjectPtr<BlockNode> new_block = CopyOnWrite(block);
      new_block->body = this->VisitStmt(new_block->body);
      return Block(std::move(new_block));
    } else if (parent_stmt->IsInstance<ForNode>()) {
      auto* loop = const_cast<ForNode*>(static_cast<const ForNode*>(parent_stmt));
      ObjectPtr<ForNode> new_loop = CopyOnWrite(loop);
      new_loop->body = this->VisitStmt(new_loop->body);
      return For(std::move(new_loop));
    }
    LOG(FATAL) << "TypeError: Unexpected type: " << parent_stmt->GetTypeKey();
    throw;
  }

  /*! \brief The `src_stmt` to be replaced */
  const StmtNode* src_stmt_;
  /*! \brief The `tgt_stmt` to be replaced in */
  const Stmt& tgt_stmt_;
  /*!
   * \brief The `seq_index` of the `src_stmt`
   * \sa StmtSRefNode
   */
  int seq_index_;
};

void ScheduleStateNode::Replace(const tir::StmtSRef& _src_sref, const Stmt& tgt_stmt,
                                const Map<Block, Block>& _block_sref_reuse) {
  if (this->debug_mode != 0) {
    const StmtNode* src_stmt = _src_sref->stmt;
    bool input_correct =
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<ForNode>()) ||
        (src_stmt->IsInstance<ForNode>() && tgt_stmt->IsInstance<BlockRealizeNode>()) ||
        (src_stmt->IsInstance<BlockNode>() && tgt_stmt->IsInstance<BlockNode>());
    if (!input_correct) {
      LOG(FATAL) << "TypeError: src_stmt has type: " << src_stmt->GetTypeKey()
                 << ". tgt_stmt has type: " << tgt_stmt->GetTypeKey() << ".\nsrc_stmt:\n"
                 << GetRef<Stmt>(src_stmt) << "\ntgt_stmt:\n"
                 << tgt_stmt;
    }
  }
  // Rule out the case that no replacement happens
  if (_src_sref->stmt == tgt_stmt.get()) {
    return;
  }
  // Reset sref as a new sref so that its content won't be affected by subsequent changes
  StmtSRef src_sref(_src_sref->stmt, _src_sref->parent, _src_sref->seq_index);
  Stmt src_stmt = GetRef<Stmt>(src_sref->stmt);
  // Step 1. Create all the nodes needed for the new sref tree.
  // After this step
  // 1) all `parent`s are correct
  // 2) all `seq_index`s are correct, except for the root
  // 3) all `stmt`s are correct, except for the root
  {
    // Step 0. Setup block_sref_reuse
    std::unordered_map<const BlockNode*, const BlockNode*> block_sref_reuse;
    block_sref_reuse.reserve(_block_sref_reuse.size() + 1);
    for (const auto& kv : _block_sref_reuse) {
      block_sref_reuse.emplace(kv.first.get(), kv.second.get());
    }
    // Step 1.1. Collect info for different kinds of reuses
    // 1) intact
    // 2) loop/block reuse
    ReuseInfo reuse_info = ReuseCollector::Collect(this, tgt_stmt);
    reuse_info.block_sref_reuse = std::move(block_sref_reuse);
    // Step 1.2. Collect loop/block reuse to their corresponding srefs
    // and remove those srefs in the `src_stmt` that are no longer used after replacement
    std::unordered_map<const Object*, StmtSRef> reused_srefs =
        SRefTreePruner::Prune(this, reuse_info, src_stmt);
    // Step 1.3. Update the sref tree, inserting newly created srefs and properly handle reused
    // srefs in `tgt_stmt`
    SRefUpdater::Update(this, src_sref->parent, reused_srefs, tgt_stmt);
  }
  // Step 2. Set the ancestors' children properly
  //   Iteratively visit the ancestors, creating new ones whose `body`s are properly fixed.
  //   The visit stops when all the ancestors are uniquely referenced, i.e. can mutate inplace.
  //   Along the way, because we create a new ancestor path,
  //   we need to update those sref points from old ancestors to newly created ones
  // Variables:
  // 1) `num_copy_steps`. The maximum number of hops until we need to copy. To reach a node that
  //   can be mutated inplace, it needs `num_copy_steps + 1` hops.
  // 2) `need_module_copy`. If true, need to mutate the PrimFunc and IRModule the sref belongs to.
  // 3) `g_var` and `g_func`. Indicate which GlobalVar and PrimFunc the sref corresponds to
  int num_copy_steps = -1;
  bool need_module_copy = false;
  const PrimFuncNode* g_func = nullptr;
  GlobalVar g_var;
  {
    int i = 0;
    const StmtSRefNode* p = src_sref.get();
    while (true) {
      if (!p->stmt->unique()) {
        num_copy_steps = i;
      }
      if (p->parent == nullptr) {
        break;
      }
      ++i;
      p = p->parent;
    }
    // Find `g_func` and `g_var` where the `src_sref` is in
    g_func = GetRootPrimFunc(this->mod, p->stmt, &g_var);
    need_module_copy = num_copy_steps == i ||             //
                       !this->mod.unique() ||             //
                       !this->mod->functions.unique() ||  //
                       !g_func->unique();
  }
  // Loop invariant:
  //
  // Before step `i`:
  // 1) `child_sref` is `src_sref` going up by `i` steps
  // 2) `child_tgt_stmt` is the subtree that `child_sref` should correspond to after replacement
  // 3) except for the subtree root, srefs that point to the subtree of `child_tgt_stmt` are
  // correct 4) for the subtree root of `child_tgt_stmt`, `child_sref` has not pointed to it yet
  // 5) `tgt_stmt` is of type Loop, Block or BlockRealize
  //
  // During step `i`:
  // 1) Create `parent_stmt` that corresponds to `child_sref->parent`
  // 2) Point `child_sref` to `child_tgt_stmt`
  // 3) `tgt_stmt` is of type Loop or Block
  StmtSRefNode* child_sref = src_sref.get();
  Stmt child_tgt_stmt = std::move(tgt_stmt);
  for (int i = 0; (need_module_copy || i <= num_copy_steps) && child_sref->parent != nullptr; ++i) {
    bool can_directly_mutate_parent = !need_module_copy && i == num_copy_steps;
    // Replace `child_sref->stmt` to `child_tgt_stmt`.
    const StmtNode* parent_stmt = child_sref->parent->stmt;
    const StmtNode* child_src_stmt = child_sref->stmt;
    // Step 2.1. Link `child_sref` to `child_tgt_stmt`
    if (i == 0) {
      // As the invariance of SRefUpdater,
      // the `seq_index` of the root of `tgt_stmt` is set as -1,
      // which might be incorrect
      SetSeqIndex(this, child_tgt_stmt, child_sref->seq_index);
    } else {
      // Point `child_sref` to `child_tgt_stmt`
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Step 2.2. Create `new_parent_stmt`, by mutating the body of `parent_stmt`
    Stmt new_parent_stmt =
        ChildReplacer::Replace(parent_stmt, child_src_stmt, child_tgt_stmt,
                               /*seq_index=*/child_sref->seq_index,
                               /*allow_copy_on_write=*/can_directly_mutate_parent);
    // Step 2.3. Go to next parent
    if (can_directly_mutate_parent) {
      // If the node can be directly mutated inplace,
      // then there is no need to update its parent and the function
      break;
    }
    child_tgt_stmt = std::move(new_parent_stmt);
    child_sref = child_sref->parent;
  }
  // Step 3. Handle the case that we mutate the root
  if (need_module_copy) {
    // From the loop invariant, upon exit, while its subtree is properly set,
    // `child_sref` is not properly to `child_tgt_stmt` yet.
    if (src_sref->parent != nullptr) {
      // Not replacing a root
      UpdateSRef(this, child_sref, child_tgt_stmt.get());
    }
    // Ensure the uniqueness of `this->mod` and `this->mod->functions`
    IRModuleNode* new_mod = this->mod.CopyOnWrite();
    MapNode* new_map = new_mod->functions.CopyOnWrite();
    // Move out the PrimFunc where the sref belong while ensuring uniqueness
    PrimFunc ref_new_func = Downcast<PrimFunc>(std::move(new_map->at(g_var)));
    ICHECK(ref_new_func.get() == g_func);
    PrimFuncNode* new_func = ref_new_func.CopyOnWrite();
    // If `g_func` was not unique, after the 3 lines above:
    //   `ref_new_func` points to a unique PrimFunc
    //   `g_func` points to the previous PrimFunc if it is not unique
    // If `g_func` was unique, after the 3 lines above:
    //   `ref_new_func` points to the same unique function that `g_func` points to
    // Update the body of the function the sref belongs to Assign
    const auto* realize = TVM_TYPE_AS(realize, g_func->body, BlockRealizeNode);
    // Make `child_tgt_stmt` the root block
    const auto* child_block = TVM_TYPE_AS(child_block, child_tgt_stmt, BlockNode);
    ObjectPtr<BlockRealizeNode> new_realize = make_object<BlockRealizeNode>(*realize);
    new_realize->block = GetRef<Block>(child_block);
    new_func->body = BlockRealize(std::move(new_realize));
    // Finally, move the `ref_new_func` back and update `this->mod`
    new_map->at(g_var) = std::move(ref_new_func);
    this->mod = GetRef<IRModule>(new_mod);
  }
  constexpr int kVerifySRefTree = static_cast<int>(ScheduleDebugMask::kVerifySRefTree);
  if (debug_mode == -1 || (debug_mode & kVerifySRefTree)) {
    VerifySRefTree(GetRef<ScheduleState>(this));
  }
}

void ScheduleStateNode::DebugVerify() const {
  constexpr int kVerifySRefTree = static_cast<int>(ScheduleDebugMask::kVerifySRefTree);
  constexpr int kVerifyAffineBinding = static_cast<int>(ScheduleDebugMask::kVerifyAffineBinding);
  constexpr int kVerifyRegionCover = static_cast<int>(ScheduleDebugMask::kVerifyRegionCover);
  constexpr int kVerifyStagePipeline = static_cast<int>(ScheduleDebugMask::kVerifyStagePipeline);
  ICHECK_GE(debug_mode, -1);
  if (debug_mode == -1 || (debug_mode & kVerifySRefTree)) {
    VerifySRefTree(GetRef<ScheduleState>(this));
  }
  if (debug_mode == -1 || (debug_mode & kVerifyAffineBinding)) {
    // TODO(@junrushao1994): Verify affine block binding
  }
  if (debug_mode == -1 || (debug_mode & kVerifyRegionCover)) {
    // TODO(@junrushao1994): Verify region cover
  }
  if (debug_mode == -1 || (debug_mode & kVerifyStagePipeline)) {
    // TODO(@junrushao1994): Verify stage pipeline
  }
}

/**************** BlockInfo-related ****************/

BlockInfo ScheduleStateNode::GetBlockInfo(const StmtSRef& block_sref) const {
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  auto it = this->block_info.find(block_sref);
  CHECK(it != this->block_info.end())
      << "IndexError: Cannot find the corresponding BlockScope to the block sref:\n"
      << GetRef<Stmt>(block_sref->stmt);
  return it->second;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(ScheduleStateNode);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleState").set_body_typed([](ObjectRef obj, int debug_mode) {
  if (const auto* func = obj.as<PrimFuncNode>()) {
    return ScheduleState(GetRef<PrimFunc>(func), debug_mode);
  }
  if (const auto* mod = obj.as<IRModuleNode>()) {
    return ScheduleState(GetRef<IRModule>(mod), debug_mode);
  }
  LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
  throw;
});
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetBlockScope")
    .set_body_method<ScheduleState>(&ScheduleStateNode::GetBlockScope);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateReplace")
    .set_body_method<ScheduleState>(&ScheduleStateNode::Replace);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleStateGetSRef")
    .set_body_typed([](ScheduleState self, Stmt stmt) -> Optional<StmtSRef> {
      auto it = self->stmt2ref.find(stmt.get());
      return it != self->stmt2ref.end() ? it->second : Optional<StmtSRef>(NullOpt);
    });

}  // namespace tir
}  // namespace tvm
