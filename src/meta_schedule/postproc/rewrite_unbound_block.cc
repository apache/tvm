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
#include <tvm/meta_schedule/schedule/cuda/thread_bind.h>

#include "../utils.h"

namespace tvm {
namespace tir {

/*! \brief Find all the blocks that are not bound */
class UnboundBlockFinder : private StmtVisitor {
 public:
  static std::vector<std::pair<StmtSRef, String>> Find(const ScheduleState& self) {
    UnboundBlockFinder finder(self);
    for (const auto& kv : self->mod->functions) {
      GlobalVar g_var = kv.first;
      BaseFunc base_func = kv.second;
      if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
        finder.global_var_name_ = g_var->name_hint;
        finder(Downcast<BlockRealize>(prim_func->body)->block->body);
      }
    }
    return std::move(finder.blocks_);
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsBlockIdx(thread_scope)) {
      ++n_block_idx_;
    } else if (IsThreadIdx(thread_scope)) {
      ++n_thread_idx_;
    }
    if (n_block_idx_ == 0 || n_thread_idx_ == 0) {
      StmtVisitor::VisitStmt_(loop);
    }
    if (IsBlockIdx(thread_scope)) {
      --n_block_idx_;
    } else if (IsThreadIdx(thread_scope)) {
      --n_thread_idx_;
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    blocks_.emplace_back(self_->stmt2ref.at(block), global_var_name_);
  }

  explicit UnboundBlockFinder(const ScheduleState& self)
      : self_{self}, blocks_{}, n_block_idx_{0}, n_thread_idx_{0} {}

  /*! \brief The schedule state */
  const ScheduleState& self_;
  /*! \brief The list of unbound blocks */
  std::vector<std::pair<StmtSRef, String>> blocks_;
  /*!  \brief The number of blockIdx above the current stmt */
  int n_block_idx_;
  /*!  \brief The number of threadIdx above the current stmt */
  int n_thread_idx_;
  /*! \brief The name of the global var */
  String global_var_name_;
};

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief Add thread binding to unbound blocks */
class RewriteUnboundBlockNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    CHECK(context->target.defined()) << "ValueError: target is not defined";
    Optional<Integer> max_threads_per_block =
        context->target.value()->GetAttr<Integer>("max_threads_per_block");
    CHECK(max_threads_per_block.defined())
        << "ValueError: missing attribute `max_threads_per_block` in the target";
    this->max_threads_per_block_ = max_threads_per_block.value().IntValue();
  }

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  Postproc Clone() const {
    ObjectPtr<RewriteUnboundBlockNode> n = make_object<RewriteUnboundBlockNode>(*this);
    return Postproc(n);
  }

 public:
  /*! \brief The max number of threads per block from Target */
  int max_threads_per_block_ = -1;
  /*! \brief The max number of threadblocks in the cuda device */
  int max_threadblocks_ = -1;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `max_threads_per_block_` is not visited
    // `max_threadblocks_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.RewriteUnboundBlock";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteUnboundBlockNode, PostprocNode);
};

bool RewriteUnboundBlockNode::Apply(const tir::Schedule& sch) {
  using tir::BlockRV;
  using tir::ExprRV;
  using tir::LoopRV;
  using tir::Schedule;
  ICHECK_NE(this->max_threads_per_block_, -1);
  auto get_factor = [t = this->max_threads_per_block_](int max_extent) -> ExprRV {
    return Integer(std::min(t, max_extent));
  };
  std::vector<std::pair<tir::StmtSRef, String>> unbound_blocks =
      tir::UnboundBlockFinder::Find(sch->state());
  for (const auto& kv : unbound_blocks) {
    tir::StmtSRef block_sref = kv.first;
    String global_var_name = kv.second;
    BlockRV block_rv = GetRVFromSRef(sch, block_sref, global_var_name);
    BindBlockThreadIdx(sch, block_rv, max_threadblocks_, max_threads_per_block_, get_factor);
  }
  return true;
}

Postproc Postproc::RewriteUnboundBlock(int max_threadblocks) {
  ObjectPtr<RewriteUnboundBlockNode> n = make_object<RewriteUnboundBlockNode>();
  n->max_threadblocks_ = max_threadblocks;
  n->max_threads_per_block_ = -1;
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteUnboundBlockNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteUnboundBlock")
    .set_body_typed(Postproc::RewriteUnboundBlock);

}  // namespace meta_schedule
}  // namespace tvm
