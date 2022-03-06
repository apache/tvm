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
  static bool Check(const Stmt& stmt) {
    try {
      ThreadExtentChecker().VisitStmt(stmt);
      return true;
    } catch (const dmlc::Error& e) {
      return false;
    }
  }

 private:
  void VisitStmt_(const ForNode* loop) {
    if (IsThreadIdx(GetThreadScope(loop))) {
      const std::string& thread_tag = loop->thread_binding.value()->thread_tag;
      if (const int64_t* p_ext = GetLoopIntExtent(loop)) {
        auto it = thread_tag2extent_.find(thread_tag);
        bool new_thread = it == thread_tag2extent_.end();
        if (new_thread) {
          thread_extent_product *= *p_ext;
          thread_tag2extent_[thread_tag] = *p_ext;
        } else {
          CHECK_EQ(it->second, *p_ext)
              << "ValueError: All loops that are bound to `" << thread_tag
              << "` should have the same extent. However, there are two loops with extent "
              << it->second << " and " << p_ext << ", which are not equal";
        }
        StmtVisitor::VisitStmt_(loop);
        if (new_thread) {
          thread_extent_product /= *p_ext;
          thread_tag2extent_.erase(thread_tag);
        }
        return;
      } else {
        throw dmlc::Error("Dynamic thread extent");
      }
    }
    StmtVisitor::VisitStmt_(loop);
  }

  void VisitStmt_(const BlockNode* block) {
    if (Optional<Integer> low_inclusive =
            GetAnn<Integer>(block, attr::meta_schedule_thread_extent_low_inclusive)) {
      if (Optional<Integer> high_inclusive =
              GetAnn<Integer>(block, attr::meta_schedule_thread_extent_high_inclusive)) {
        int64_t low = low_inclusive.value()->value;
        int64_t high = high_inclusive.value()->value;
        if (!(low <= thread_extent_product && thread_extent_product <= high)) {
          throw dmlc::Error("Thread extent");
        }
      }
    }
    StmtVisitor::VisitStmt_(block);
  }

  int64_t thread_extent_product = 1;

  /*! \brief A mapping from a thread tag to its thread extent */
  std::unordered_map<std::string, int64_t> thread_tag2extent_;
};
}  // namespace tir
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
  Map<String, PrimExpr> target_constraints_{nullptr};

  void InitializeWithTuneContext(const TuneContext& context) final {
    ICHECK(context->target.defined());
    Target target = context->target.value();
    this->target_constraints_ = Map<String, PrimExpr>{
        {"max_shared_memory_per_block", Extract(target, "max_shared_memory_per_block")},
        {"max_threads_per_block", Extract(target, "max_threads_per_block")},
        {"max_vthread", Integer(8)},
        {"max_vector_bytes", Integer(16)}};
  }

  bool Verify(const IRModule& mod) const {
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<tir::PrimFuncNode>()) {
        if (!tir::VerifyGPUCode(GetRef<tir::PrimFunc>(prim_func), this->target_constraints_)) {
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
        if (!tir::ThreadExtentChecker::Check(prim_func->body)) {
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
          pass_list.push_back(tir::transform::UnifyThreadBinding());
          pass_list.push_back(tir::transform::CompactBufferAllocation());
          pass_list.push_back(tir::transform::LowerMatchBuffer());
          pass_list.push_back(tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(tir::transform::FlattenBuffer());
          pass_list.push_back(tir::transform::BF16Legalize());
          pass_list.push_back(tir::transform::NarrowDataType(32));
          pass_list.push_back(tir::transform::Simplify());

          // Phase 2
          pass_list.push_back(tir::transform::VectorizeLoop(true));
          pass_list.push_back(tir::transform::InjectVirtualThread());
          pass_list.push_back(tir::transform::InjectDoubleBuffer());
          pass_list.push_back(tir::transform::StorageRewrite());
          pass_list.push_back(tir::transform::MergeDynamicSharedMemoryAllocations());

          // Convert Function to IRModule
          transform::PassContext pass_ctx = transform::PassContext::Current();
          tir::PrimFunc f = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol",
                                     runtime::String(g_var->name_hint));
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
