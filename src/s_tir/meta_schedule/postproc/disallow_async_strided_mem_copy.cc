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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief Check if an IRModule has any async strided mem copies. */
struct AsyncStridedMemCopyFinder : public StmtExprVisitor {
  using StmtExprVisitor::Visit_;

  static bool Find(const IRModule& mod) {
    auto finder = ffi::make_object<AsyncStridedMemCopyFinder>();
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<PrimFuncNode>()) {
        finder->Visit(prim_func->body);
        if (finder->found_) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForNode* loop) final {
    if (!found_) {
      input_iters.Set(loop->loop_var, Range(loop->min, loop->extent));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op) final {
    bool previous = in_async_copy_;
    in_async_copy_ |= op->op.same_as(s_tir::async_copy_scope());
    auto result = StmtExprVisitor::Visit_(op);
    in_async_copy_ = previous;
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* op) final {
    if (!found_ && in_async_copy_) {
      if (const auto* load = op->value.as<TensorLoadNode>()) {
        // Inspect each copy, including copies grouped under one commit or predicate.
        sym::Analyzer analyzer;
        auto store_map = DetectIterMap(op->indices, input_iters, 1, sym::IterMapLevel::Surjective,
                                       analyzer, false);
        auto load_map = DetectIterMap(load->indices, input_iters, 1, sym::IterMapLevel::Surjective,
                                      analyzer, false);
        found_ = !store_map->errors.empty() || !load_map->errors.empty();
      }
    }
    return std::nullopt;
  }

  bool in_async_copy_ = false;
  bool found_ = false;
  ffi::Map<PrimVar, Range> input_iters = ffi::Map<PrimVar, Range>();
};

}  // namespace s_tir

namespace s_tir {
namespace meta_schedule {

/*! \brief Check if the IRModule has any loop with non-constant extent. */
class DisallowAsyncStridedMemCopyNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {
    /* Null check */
    TVM_FFI_ICHECK(context->target) << "Context must contain a target";
    this->target = context->target.value();
  }
  // Inherited from PostprocNode
  bool Apply(const s_tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* prim_func = base_func.as<tirx::PrimFuncNode>()) {
        IRModule lowered{ffi::UnsafeInit()};
        try {
          auto pass_list = ffi::Array<tvm::transform::Pass>();
          pass_list.push_back(tirx::transform::BindTarget(this->target));
          pass_list.push_back(s_tir::transform::LowerInitBlock());
          pass_list.push_back(s_tir::transform::PlanAndUpdateBufferAllocationLocation());
          pass_list.push_back(s_tir::transform::ConvertBlocksToOpaque());
          pass_list.push_back(s_tir::transform::CompactBufferAllocation());
          pass_list.push_back(s_tir::transform::LowerMatchBuffer());
          pass_list.push_back(s_tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(s_tir::transform::LowerOpaqueBlock());
          pass_list.push_back(s_tir::transform::LowerThreadBinding());
          pass_list.push_back(tirx::transform::FlattenBuffer());
          pass_list.push_back(tirx::transform::BF16ComputeLegalize());
          pass_list.push_back(tirx::transform::NarrowDataType(32));
          pass_list.push_back(tirx::transform::StmtSimplify());
          pass_list.push_back(s_tir::transform::InjectVirtualThread());
          pass_list.push_back(tirx::transform::VectorizeLoop(true));
          pass_list.push_back(tirx::transform::StorageRewrite());
          tirx::PrimFunc f = WithAttr(ffi::GetRef<tirx::PrimFunc>(prim_func), "global_symbol",
                                      ffi::String(g_var->name_hint));
          IRModule mod =
              IRModule(ffi::Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
        } catch (const std::runtime_error& e) {
          return false;
        }
        if (s_tir::AsyncStridedMemCopyFinder::Find(lowered)) {
          return false;
        }
      }
    }
    return true;
  }
  // Inherited from PostprocNode
  Postproc Clone() const {
    ffi::ObjectPtr<DisallowAsyncStridedMemCopyNode> n =
        ffi::make_object<DisallowAsyncStridedMemCopyNode>(*this);
    return Postproc(n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DisallowAsyncStridedMemCopyNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.DisallowAsyncStridedMemCopy",
                                    DisallowAsyncStridedMemCopyNode, PostprocNode);

 private:
  tvm::Target target;
};

Postproc Postproc::DisallowAsyncStridedMemCopy() {
  ffi::ObjectPtr<DisallowAsyncStridedMemCopyNode> n =
      ffi::make_object<DisallowAsyncStridedMemCopyNode>();
  return Postproc(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DisallowAsyncStridedMemCopyNode::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocDisallowAsyncStridedMemCopy",
                        Postproc::DisallowAsyncStridedMemCopy);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
