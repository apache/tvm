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

/*! \brief The rewrite type for an unbound block */
enum class BindType : int32_t {
  /*! \brief No additional thread binding is needed */
  kNoBind = 0,
  /*! \brief Need to bind to blockIdx */
  kBindBlock = 1,
  /*! \brief Need to bind to both blockIdx and threadIdx */
  kBindBlockThread = 2,
};

/*!
 * \brief Check the combination of bindings to be added to the block
 * \param block_sref The block to be checked
 * \param fuse_first_num The number of loops to be fused
 * \return The type of binding to be added to the block
 */
BindType GetBindType(const StmtSRef& block_sref, int* fuse_first_num) {
  Array<StmtSRef> loops = tir::GetLoops(block_sref);
  int n = loops.size();
  if (n == 0) {
    return BindType::kNoBind;
  }
  int i_block_idx = -1;
  int i_thread_idx = -1;
  int i_multi_child = -1;
  int i_spatial_loop = -1;
  for (int i = 0; i < n; ++i) {
    const StmtSRef& loop_sref = loops[i];
    const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsBlockIdx(thread_scope)) {
      if (i_block_idx == -1) {
        i_block_idx = i;
      }
    }
    if (IsThreadIdx(thread_scope)) {
      if (i_thread_idx == -1) {
        i_thread_idx = i;
      }
    }
    if (loop->kind != tir::ForKind::kSerial) {
      if (i_multi_child == -1) {
        i_multi_child = i;
      }
    }
    if (!IsSingleStmt(loop->body)) {
      if (i_multi_child == -1) {
        i_multi_child = i + 1;
      }
    }
    if (tir::GetLoopIterType(loop_sref) == IterVarType::kDataPar) {
      if (i_spatial_loop == i - 1) {
        ++i_spatial_loop;
      }
    }
  }
  if (i_multi_child == -1) {
    i_multi_child = n;
  }
  if ((i_block_idx != -1 && i_thread_idx != -1) || i_spatial_loop == -1) {
    return BindType::kNoBind;
  } else if (i_block_idx != -1 && i_thread_idx == -1) {
    ICHECK(false) << "Unsupported case, where blockIdx is bound but threadIdx is not";
    throw;
  } else if (i_block_idx == -1 && i_thread_idx != -1) {
    *fuse_first_num = std::min(std::min(i_multi_child, i_thread_idx), i_spatial_loop + 1);
    return BindType::kBindBlock;
  } else {  // i_block_idx == -1 && i_thread_idx == -1
    *fuse_first_num = std::min(i_multi_child, i_spatial_loop + 1);
    return BindType::kBindBlockThread;
  }
}

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
    Optional<Integer> max_num_threads =
        context->target.value()->GetAttr<Integer>("max_threads_per_block");
    CHECK(max_num_threads.defined())
        << "ValueError: missing attribute `max_threads_per_block` in the target";
    this->max_num_threads_ = max_num_threads.value();
  }

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

 public:
  /*! \brief The max number of threads per block from Target */
  int max_num_threads_ = -1;
  /*! \brief The max number of threadblocks in the cuda device */
  int max_threadblock_ = -1;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `max_num_threads_` is not visited
    // `max_threadblock_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.RewriteUnboundBlock";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteUnboundBlockNode, PostprocNode);
};

bool RewriteUnboundBlockNode::Apply(const tir::Schedule& sch) {
  using tir::BlockRV;
  using tir::LoopRV;
  using tir::Schedule;
  ICHECK_NE(this->max_num_threads_, -1);
  std::vector<std::pair<tir::StmtSRef, String>> unbound_blocks =
      tir::UnboundBlockFinder::Find(sch->state());
  for (const auto& kv : unbound_blocks) {
    tir::StmtSRef block_sref = kv.first;
    String global_var_name = kv.second;
    int fuse_first_num = 0;
    tir::BindType bind_type = tir::GetBindType(block_sref, &fuse_first_num);
    if (bind_type == tir::BindType::kNoBind) {
      continue;
    }
    BlockRV block_rv = GetRVFromSRef(sch, block_sref, global_var_name);
    Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    LoopRV fused = sch->Fuse({loop_rvs.begin(), loop_rvs.begin() + fuse_first_num});
    if (bind_type == tir::BindType::kBindBlock) {
      sch->Bind(fused, "blockIdx.x");
    } else if (bind_type == tir::BindType::kBindBlockThread) {
      int64_t extent_size = 0;
      Array<LoopRV> splits;
      if (const int64_t* extent_ptr = tir::GetLoopIntExtent(sch->Get(fused).get())) {
        extent_size = *extent_ptr;
        if (extent_size > max_threadblock_ * max_num_threads_) {
          splits =
              sch->Split(fused, {NullOpt, Integer(max_threadblock_), Integer(max_num_threads_)});
          ICHECK_EQ(splits.size(), 3);
          sch->Reorder({splits[1], splits[2], splits[0]});
          sch->Bind(splits[1], "blockIdx.x");
          sch->Bind(splits[2], "threadIdx.x");
        } else {
          ICHECK_NE(extent_size, 0);
          splits = sch->Split(
              fused,
              {NullOpt, Integer(std::min(static_cast<int64_t>(max_num_threads_), extent_size))});
          ICHECK_EQ(splits.size(), 2);
          sch->Bind(splits[0], "blockIdx.x");
          sch->Bind(splits[1], "threadIdx.x");
        }
      } else {
        // loop is dynamic, returns nullptr
        splits = sch->Split(fused, {NullOpt, Integer(max_num_threads_)});
        ICHECK_EQ(splits.size(), 2);
        sch->Bind(splits[0], "blockIdx.x");
        sch->Bind(splits[1], "threadIdx.x");
      }
    }
  }
  return true;
}

Postproc Postproc::RewriteUnboundBlock(int max_threadblock) {
  ObjectPtr<RewriteUnboundBlockNode> n = make_object<RewriteUnboundBlockNode>();
  n->max_threadblock_ = max_threadblock;
  n->max_num_threads_ = -1;
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteUnboundBlockNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteUnboundBlock")
    .set_body_typed(Postproc::RewriteUnboundBlock);

}  // namespace meta_schedule
}  // namespace tvm
