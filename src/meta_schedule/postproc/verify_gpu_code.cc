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
#include <tvm/tir/transform.h>

#include "../utils.h"

namespace tvm {
namespace tir {

class ThreadExtentChecker : private StmtVisitor {
 public:
  static bool Check(const Stmt& stmt, int thread_warp_size) {
    try {
      ICHECK(thread_warp_size > 0);
      ThreadExtentChecker checker(thread_warp_size);
      checker.VisitStmt(stmt);
      return true;
    } catch (const dmlc::Error& e) {
      return false;
    }
  }

 private:
  explicit ThreadExtentChecker(int thread_warp_size) : thread_warp_size_(thread_warp_size) {}

  void VisitStmt_(const ForNode* loop) {
    runtime::ThreadScope thread_scope = GetThreadScope(loop);
    if (IsThreadIdx(thread_scope)) {
      if (const int64_t* p_ext = GetLoopIntExtent(loop)) {
        int64_t ext = *p_ext;
        if (thread_scope.dim_index == 0) {
          std::swap(thread_idx_x, ext);
          StmtVisitor::VisitStmt_(loop);
          std::swap(thread_idx_x, ext);
        } else if (thread_scope.dim_index == 1) {
          std::swap(thread_idx_y, ext);
          StmtVisitor::VisitStmt_(loop);
          std::swap(thread_idx_y, ext);
        } else if (thread_scope.dim_index == 2) {
          std::swap(thread_idx_z, ext);
          StmtVisitor::VisitStmt_(loop);
          std::swap(thread_idx_z, ext);
        } else {
          StmtVisitor::VisitStmt_(loop);
        }
        return;
      } else {
        throw dmlc::Error("Dynamic thread extent");
      }
    }
    StmtVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const BlockNode* block) {
    int old_thread_idx_x = thread_idx_x;
    if (block->annotations.count(attr::warp_execution)) {
      thread_idx_x = thread_warp_size_;
    }
    if (Optional<Integer> low_inclusive =
            GetAnn<Integer>(block, attr::meta_schedule_thread_extent_low_inclusive)) {
      if (Optional<Integer> high_inclusive =
              GetAnn<Integer>(block, attr::meta_schedule_thread_extent_high_inclusive)) {
        int64_t low = low_inclusive.value()->value;
        int64_t high = high_inclusive.value()->value;
        int64_t thread_extent_product = thread_idx_x * thread_idx_y * thread_idx_z;
        if (!(low <= thread_extent_product && thread_extent_product <= high)) {
          throw dmlc::Error("Thread extent");
        }
      }
    }
    StmtVisitor::VisitStmt_(block);
    thread_idx_x = old_thread_idx_x;
  }

  int64_t thread_idx_x = 1;
  int64_t thread_idx_y = 1;
  int64_t thread_idx_z = 1;
  int thread_warp_size_ = -1;
};

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief Extract attribute from a target. */
Integer Extract(const Target& target, const char* name) {
  ICHECK(target.defined());
  if (Optional<Integer> v = target->GetAttr<Integer>(name)) {
    return v.value();
  }
  LOG(FATAL) << "AttributedError: \"" << name << "\" is not defined in the target";
  throw;
}

/*! \brief Verify the correctness of the generated GPU code. */
class VerifyGPUCodeNode : public PostprocNode {
 public:
  Target target_{nullptr};
  Map<String, PrimExpr> target_constraints_{nullptr};
  int thread_warp_size_ = -1;

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    this->target_ = context->target.value();
    this->target_constraints_ = Map<String, PrimExpr>{
        {"max_shared_memory_per_block", Extract(this->target_, "max_shared_memory_per_block")},
        {"max_threads_per_block", Extract(this->target_, "max_threads_per_block")},
        {"max_vthread", Integer(8)},
        {"max_vector_bytes", Integer(16)},
    };
    thread_warp_size_ = Extract(this->target_, "thread_warp_size").IntValue();
  }

  bool Verify(const IRModule& mod) const {
    for (const auto& kv : mod->functions) {
      if (auto prim_func = kv.second.as<tir::PrimFunc>()) {
        if (!tir::VerifyGPUCode(prim_func.value(), this->target_constraints_)) {
          return false;
        }
      }
    }
    return true;
  }

  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
        if (!tir::ThreadExtentChecker::Check(prim_func->body, thread_warp_size_)) {
          return false;
        }
        IRModule lowered{nullptr};
        try {
          auto pass_list = Array<tvm::transform::Pass>();
          // Phase 1
          // First three passes are not needed in TIR schedule.
          // pass_list.push_back(tir::transform::InjectPrefetch());
          // pass_list.push_back(tir::transform::TextureFlatten());
          // pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
          pass_list.push_back(tir::transform::LowerCrossThreadReduction());
          pass_list.push_back(tir::transform::LowerInitBlock());
          pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
          pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
          pass_list.push_back(tir::transform::LiftThreadBinding());
          pass_list.push_back(tir::transform::ManifestSharedMemoryLocalStage());
          pass_list.push_back(tir::transform::CompactBufferAllocation());
          pass_list.push_back(tir::transform::Simplify());
          pass_list.push_back(tir::transform::LowerAutoCopy());
          pass_list.push_back(tir::transform::UnifyThreadBinding());
          pass_list.push_back(tir::transform::LowerMatchBuffer());
          pass_list.push_back(tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(tir::transform::LowerOpaqueBlock());
          pass_list.push_back(tir::transform::FlattenBuffer());
          pass_list.push_back(tir::transform::BF16ComputeLegalize());
          pass_list.push_back(tir::transform::NarrowDataType(32));
          pass_list.push_back(tir::transform::Simplify());
          // Phase 2
          pass_list.push_back(tir::transform::VectorizeLoop(true));
          pass_list.push_back(tir::transform::InjectVirtualThread());
          pass_list.push_back(tir::transform::InjectDoubleBuffer());
          pass_list.push_back(tir::transform::StorageRewrite());
          pass_list.push_back(tir::transform::MergeSharedMemoryAllocations());
          pass_list.push_back(tir::transform::LowerIntrin());
          // Convert Function to IRModule
          transform::PassContext pass_ctx = transform::PassContext::Current();
          tir::PrimFunc f = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol",
                                     runtime::String(g_var->name_hint));
          f = WithAttr(f, tvm::attr::kTarget, this->target_);  // Required for LowerIntrin
          bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
          if (noalias) {
            f = WithAttr(std::move(f), "tir.noalias", Bool(true));
          }
          IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
        } catch (const dmlc::Error& e) {
          return false;
        }
        if (!Verify(lowered)) {
          return false;
        }
      }
    }
    return true;
  }

  Postproc Clone() const {
    ObjectPtr<VerifyGPUCodeNode> n = make_object<VerifyGPUCodeNode>(*this);
    n->target_constraints_ = this->target_constraints_;
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.VerifyGPUCode";
  TVM_DECLARE_FINAL_OBJECT_INFO(VerifyGPUCodeNode, PostprocNode);
};

Postproc Postproc::VerifyGPUCode() {
  ObjectPtr<VerifyGPUCodeNode> n = make_object<VerifyGPUCodeNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(VerifyGPUCodeNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocVerifyGPUCode").set_body_typed(Postproc::VerifyGPUCode);

}  // namespace meta_schedule
}  // namespace tvm
