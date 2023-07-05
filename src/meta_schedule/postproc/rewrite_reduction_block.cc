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

/*! \brief The visitor that finds all the reduction block to be decomposed */
struct ReductionBlockFinder : private StmtVisitor {
 public:
  /*! \brief Find all the reduction blocks that should be decomposed */
  static std::vector<std::pair<StmtSRef, String>> Find(const ScheduleState& self) {
    std::vector<std::pair<StmtSRef, String>> results;
    for (const auto& kv : self->mod->functions) {
      GlobalVar g_var = kv.first;
      BaseFunc base_func = kv.second;
      if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
        ReductionBlockFinder finder;
        finder(prim_func->body);
        for (const BlockNode* block : finder.results_) {
          results.emplace_back(self->stmt2ref.at(block), g_var->name_hint);
        }
      }
    }
    return results;
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsThreadIdx(thread_scope) || IsBlockIdx(thread_scope)) {
      thread_bound_loop_vars_.insert(loop->loop_var.get());
    }
    StmtVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    if (realize->block->init.defined() && AllReductionIterVarAreUnbound(realize)) {
      results_.push_back(realize->block.get());
    }
    StmtVisitor::VisitStmt_(realize);
  }

  bool AllReductionIterVarAreUnbound(const BlockRealizeNode* realize) const {
    if (thread_bound_loop_vars_.empty()) {
      return true;
    }
    auto f_find = [this](const VarNode* var) -> bool { return thread_bound_loop_vars_.count(var); };
    const BlockNode* block = realize->block.get();
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n = block->iter_vars.size();
    for (int i = 0; i < n; ++i) {
      IterVar iter_var = block->iter_vars[i];
      PrimExpr binding = realize->iter_values[i];
      if (iter_var->iter_type == tir::kCommReduce) {
        if (UsesVar(binding, f_find)) {
          return false;
        }
      }
    }
    return true;
  }

  /*! \brief The results of the collection */
  std::vector<const BlockNode*> results_;
  /*! \brief Loop variables that are bound to threads */
  std::unordered_set<const VarNode*> thread_bound_loop_vars_;
};

/*!
 * \brief Find the innermost loop that the `init` of the input block could be decomposed to
 * \param block_sref The StmtSRef of the block to be decomposed
 * \return The index of the innermost loop where the `init` of the input block could be decomposed,
 * or -1 if the `init` does not need to be decomposed.
 */
int FindDecomposePoint(const StmtSRef& block_sref) {
  Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  int n = loop_srefs.size();
  for (int i = 0; i < n; ++i) {
    if (GetLoopIterType(loop_srefs[i]) != IterVarType::kDataPar) {
      return i;
    }
  }
  return -1;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief Rewrite reduction block by moving the init block out */
class RewriteReductionBlockNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  Postproc Clone() const {
    ObjectPtr<RewriteReductionBlockNode> n = make_object<RewriteReductionBlockNode>(*this);
    return Postproc(n);
  }

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.RewriteReductionBlock";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteReductionBlockNode, PostprocNode);
};

bool RewriteReductionBlockNode::Apply(const tir::Schedule& sch) {
  for (;;) {
    std::vector<std::pair<tir::StmtSRef, String>> results =
        tir::ReductionBlockFinder::Find(sch->state());
    int rewritten = 0;
    for (const auto& kv : results) {
      const tir::StmtSRef& block_sref = kv.first;
      const String& global_var_name = kv.second;
      int decompose_point = tir::FindDecomposePoint(block_sref);
      if (decompose_point == -1) {
        continue;
      }
      tir::BlockRV block_rv = GetRVFromSRef(sch, block_sref, global_var_name);
      Array<tir::LoopRV> loop_rvs = sch->GetLoops(block_rv);
      tir::BlockRV init_block_rv = sch->DecomposeReduction(block_rv, loop_rvs[decompose_point]);

      // Rewrite auto tensorization related annotations
      if (tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_auto_tensorize).defined()) {
        // Remove tensorization annotation as it shouldn't be propagated to the init block.
        sch->Unannotate(init_block_rv, tir::attr::meta_schedule_auto_tensorize);
        Optional<String> tensorize_init =
            tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_auto_tensorize_init);
        // The annotation of tensorization of the init statement should be moved to the init block
        // after 'DecomposeReduction'.
        // Annotate to hint `RewriteTensorize` postprocessor even if tensorize_init is NullOpt.
        sch->Annotate(init_block_rv, tir::attr::meta_schedule_auto_tensorize,
                      tensorize_init.value_or(""));
        if (tensorize_init.defined()) {
          sch->Unannotate(block_rv, tir::attr::meta_schedule_auto_tensorize_init);
          sch->Unannotate(init_block_rv, tir::attr::meta_schedule_auto_tensorize_init);
        }
      }
      ++rewritten;
    }
    if (rewritten == 0) {
      break;
    }
  }
  return true;
}

Postproc Postproc::RewriteReductionBlock() {
  ObjectPtr<RewriteReductionBlockNode> n = make_object<RewriteReductionBlockNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteReductionBlockNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteReductionBlock")
    .set_body_typed(Postproc::RewriteReductionBlock);

}  // namespace meta_schedule
}  // namespace tvm
