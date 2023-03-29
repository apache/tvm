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

/*! \brief Check if an IRModule has any async strided mem copies. */
struct AsyncStridedMemCopyFinder : private StmtExprVisitor {
 public:
  static bool Find(const IRModule& mod) {
    AsyncStridedMemCopyFinder finder;
    for (const auto& kv : mod->functions) {
      if (const auto* prim_func = kv.second.as<PrimFuncNode>()) {
        finder(prim_func->body);
        if (finder.found_) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    if (!found_) {
      input_iters.Set(loop->loop_var, Range(loop->min, loop->extent));
      StmtExprVisitor::VisitStmt_(loop);
    }
  }

  void VisitStmt_(const AttrStmtNode* attrStmt) final {
    if (!found_) {
      if (attrStmt->attr_key == tir::attr::async_commit_queue_scope) {
        auto async_scope = attrStmt->body.as<AttrStmtNode>();
        if (!async_scope) {
          StmtExprVisitor::VisitStmt_(attrStmt);
        }

        auto for_loop = async_scope->body.as<ForNode>();
        if (!for_loop) {
          StmtExprVisitor::VisitStmt_(attrStmt);
        }

        input_iters.Set(for_loop->loop_var, Range(for_loop->min, for_loop->extent));

        auto bufferstorenode = for_loop->body.as<BufferStoreNode>();
        if (!bufferstorenode) {
          StmtExprVisitor::VisitStmt_(attrStmt);
        }

        auto bufferloadnode = bufferstorenode->value.as<BufferLoadNode>();
        if (!bufferloadnode) {
          StmtExprVisitor::VisitStmt_(attrStmt);
        }

        // get store buffer; assert it exists and is contiguous given it uses a single index
        auto bufferstore = bufferstorenode->buffer.as<BufferNode>();

        // get load buffer; assert it exists and is contiguous given it uses a single index
        auto bufferload = bufferloadnode->buffer.as<BufferNode>();

        if (!bufferstore || !bufferload) {
          StmtExprVisitor::VisitStmt_(attrStmt);
        }

        // map loop variable to zero for the store index & simplify
        Array<PrimExpr> store_index = bufferstorenode->indices;

        // Use DetectIterMap to detect whether store index is non-contiguous.
        arith::Analyzer analyzer;
        auto store_iter_map = DetectIterMap(store_index, input_iters, 1,
                                            arith::IterMapLevel::Surjective, &analyzer, false);
        if (!store_iter_map->errors.empty()) {
          found_ = true;
        }

        // map loop variable to zero for the load index & simplify
        Array<PrimExpr> load_index = bufferloadnode->indices;

        // Use DetectIterMap to detect whether load index is non-contiguous.
        auto load_iter_map = DetectIterMap(load_index, input_iters, 1,
                                           arith::IterMapLevel::Surjective, &analyzer, false);
        if (!load_iter_map->errors.empty()) {
          found_ = true;
        }
      }
      if (!found_) {
        StmtExprVisitor::VisitStmt_(attrStmt);
      }
    }
  }

  bool found_ = false;
  Map<Var, Range> input_iters = Map<Var, Range>();
};

}  // namespace tir

namespace meta_schedule {

/*! \brief Check if the IRModule has any loop with non-constant extent. */
class DisallowAsyncStridedMemCopyNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
        IRModule lowered{nullptr};
        try {
          auto pass_list = Array<tvm::transform::Pass>();
          pass_list.push_back(tir::transform::LowerInitBlock());
          pass_list.push_back(tir::transform::PlanAndUpdateBufferAllocationLocation());
          pass_list.push_back(tir::transform::ConvertBlocksToOpaque());
          pass_list.push_back(tir::transform::CompactBufferAllocation());
          pass_list.push_back(tir::transform::LowerMatchBuffer());
          pass_list.push_back(tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(tir::transform::LowerOpaqueBlock());
          pass_list.push_back(tir::transform::FlattenBuffer());
          pass_list.push_back(tir::transform::BF16ComputeLegalize());
          pass_list.push_back(tir::transform::NarrowDataType(32));
          pass_list.push_back(tir::transform::Simplify());
          pass_list.push_back(tir::transform::InjectVirtualThread());
          pass_list.push_back(tir::transform::InjectDoubleBuffer());
          pass_list.push_back(tir::transform::VectorizeLoop(true));
          pass_list.push_back(tir::transform::StorageRewrite());
          tir::PrimFunc f = WithAttr(GetRef<tir::PrimFunc>(prim_func), "global_symbol",
                                     runtime::String(g_var->name_hint));
          IRModule mod = IRModule(Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
        } catch (const dmlc::Error& e) {
          return false;
        }
        if (tir::AsyncStridedMemCopyFinder::Find(lowered)) {
          return false;
        }
      }
    }
    return true;
  }
  // Inherited from PostprocNode
  Postproc Clone() const {
    ObjectPtr<DisallowAsyncStridedMemCopyNode> n =
        make_object<DisallowAsyncStridedMemCopyNode>(*this);
    return Postproc(n);
  }

  static constexpr const char* _type_key = "meta_schedule.DisallowAsyncStridedMemCopy";
  TVM_DECLARE_FINAL_OBJECT_INFO(DisallowAsyncStridedMemCopyNode, PostprocNode);
};

Postproc Postproc::DisallowAsyncStridedMemCopy() {
  ObjectPtr<DisallowAsyncStridedMemCopyNode> n = make_object<DisallowAsyncStridedMemCopyNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(DisallowAsyncStridedMemCopyNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocDisallowAsyncStridedMemCopy")
    .set_body_typed(Postproc::DisallowAsyncStridedMemCopy);

}  // namespace meta_schedule
}  // namespace tvm
