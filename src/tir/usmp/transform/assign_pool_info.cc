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

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/algorithms.h>
#include <tvm/tir/usmp/analysis.h>
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>
#include <string>

namespace tvm {
namespace tir {
namespace usmp {

/*! \brief Assign PoolInfo objects to allocate that does not have any.
 * The schedulers have the oppurtunity to assign PoolInfo objects to
 * allocate nodes. However, each allocate node is expected to have
 * at least one PoolInfo node assigned to it. If it was not the case,
 * this Pass will assign all PoolInfo objects that the target could
 * access.*/
class PoolInfoAssigner : public StmtExprMutator {
 public:
  explicit PoolInfoAssigner(const IRModule& module) {
    PrimFunc main_func =
        Downcast<PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
    ICHECK(main_func.defined()) << "main function is not in the module";
    Optional<Target> target_host = main_func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target_host) << "main function does not have a target attr";
    WorkspaceMemoryPools workspace_pools =
        module->GetAttr<WorkspaceMemoryPools>(tvm::attr::kWorkspaceMemoryPools)
            .value_or(WorkspaceMemoryPools({CreateDefaultWorkspaceMemoryPool(module)}));
    // make default ConstantPoolInfo if no constant and no workspace pool infos supplied
    ConstantMemoryPools constant_pools =
        module->GetAttr<ConstantMemoryPools>(tvm::attr::kConstantMemoryPools)
            .value_or(
                module->GetAttr<WorkspaceMemoryPools>(tvm::attr::kWorkspaceMemoryPools).defined()
                    ? ConstantMemoryPools()
                    : ConstantMemoryPools({CreateDefaultConstantMemoryPool(module)}));
    auto to_map = [](auto pool_infos) {
      Map<String, Array<PoolInfo>> pool_map;
      for (const PoolInfo& pool_info : pool_infos) {
        for (const auto& tgt : pool_info->targets) {
          if (pool_map.find(tgt->str()) == pool_map.end()) {
            pool_map.Set(tgt->str(), Array<PoolInfo>());
          }
          Array<PoolInfo> pool_info_arr = pool_map[tgt->str()];
          pool_info_arr.push_back(pool_info);
          pool_map.Set(tgt->str(), pool_info_arr);
        }
      }
      return pool_map;
    };

    target_pool_infos_ = to_map(workspace_pools->pools);
    if (constant_pools.defined()) {
      target_const_pool_infos_ = to_map(constant_pools->pools);
    }
    mod_ = module->ShallowCopy();
  }

  IRModule operator()();

 private:
  Stmt VisitStmt_(const AllocateNode* op) override;
  Stmt VisitStmt_(const AllocateConstNode* op) override;

  IRModule mod_;
  Map<String, Array<PoolInfo>> target_pool_infos_;
  Map<String, Array<PoolInfo>> target_const_pool_infos_;
  PrimFunc func_;
  WorkspacePoolInfo CreateDefaultWorkspaceMemoryPool(const IRModule& module);
  ConstantPoolInfo CreateDefaultConstantMemoryPool(const IRModule& module) {
    auto p = CreateDefaultWorkspaceMemoryPool(module);
    return ConstantPoolInfo(
        "global_const_workspace", {p->targets}, {},
        PoolInfoProperties(kUnrestrictedPoolSizeHint, kUnknownClockFrequency, kUnknownReadBandwidth,
                           kUnknownWriteBandwidth, 0, 0, {p->target_burst_bytes}, Bool(true)));
  }
};

WorkspacePoolInfo PoolInfoAssigner::CreateDefaultWorkspaceMemoryPool(const tvm::IRModule& module) {
  VLOG(1) << "Creating default memory pool for:" << std::endl << module;
  Map<Target, String> target_access;
  tir::PrimFunc tir_main_func =
      Downcast<tir::PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
  Target target_host = tir_main_func->GetAttr<Target>(tvm::attr::kTarget).value();
  for (const auto& kv : module->functions) {
    BaseFunc func = kv.second;
    Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
    target_access.Set(target.value_or(target_host), kTargetPoolReadWriteAccess);
  }
  Array<Target> targets;
  for (const auto& kv : target_access) {
    bool exist = false;
    // Exclude targets with the same string representation
    for (const auto& t : targets) {
      if (t->str() == kv.first->str()) {
        exist = true;
      }
    }
    if (!exist) {
      targets.push_back(kv.first);
    }
  }
  return WorkspacePoolInfo(
      "global_workspace", targets,
      PoolInfoProperties(kUnrestrictedPoolSizeHint, kUnknownClockFrequency, kUnknownReadBandwidth,
                         kUnknownWriteBandwidth, 0, 0, {{target_host, 1}}, Bool(true)));
}

Stmt PoolInfoAssigner::VisitStmt_(const AllocateNode* op) {
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following PrimFunc does not have a target attr: \n" << func_;
  Map<String, ObjectRef> annotations = Map<String, ObjectRef>(op->annotations);
  if (op->annotations.find(kPoolCandidatesAllocateAttr) == op->annotations.end()) {
    ICHECK(target_pool_infos_.count(tgt.value()->str()) > 0)
        << "Target " << tgt << " not found among " << target_pool_infos_;
    annotations.Set(kPoolCandidatesAllocateAttr, target_pool_infos_[tgt.value()->str()]);
  }
  Stmt body = VisitStmt(op->body);
  auto allocate =
      Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body, annotations);
  return std::move(allocate);
}

Stmt PoolInfoAssigner::VisitStmt_(const AllocateConstNode* op) {
  if (!target_const_pool_infos_.size()) {
    return StmtExprMutator::VisitStmt_(op);
  }
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following PrimFunc does not have a target attr: \n" << func_;
  Map<String, ObjectRef> annotations = Map<String, ObjectRef>(op->annotations);
  if (op->annotations.find(kPoolCandidatesAllocateAttr) == op->annotations.end()) {
    annotations.Set(kPoolCandidatesAllocateAttr, target_const_pool_infos_[tgt.value()->str()]);
    annotations.Set(kTargetPoolReadOnlyAccess, Integer(1));
  }
  Stmt body = VisitStmt(op->body);
  auto allocate_const =
      AllocateConst(op->buffer_var, op->dtype, op->extents, op->data, body, annotations);
  return std::move(allocate_const);
}

IRModule PoolInfoAssigner::operator()() {
  for (const auto& kv : mod_->functions) {
    GlobalVar gv = kv.first;
    if (kv.second->IsInstance<PrimFuncNode>()) {
      func_ = Downcast<PrimFunc>(kv.second);
      Stmt body = this->VisitStmt(func_->body);
      PrimFunc new_prim_func =
          PrimFunc(func_->params, body, func_->ret_type, func_->buffer_map, func_->attrs);
      mod_->Update(gv, new_prim_func);
    }
  }
  return mod_;
}

namespace transform {

tvm::transform::Pass AssignPoolInfo() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return PoolInfoAssigner(m)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.usmp.AssignPoolInfo", {});
}

TVM_REGISTER_GLOBAL("tir.usmp.transform.AssignPoolInfo").set_body_typed(AssignPoolInfo);

}  // namespace transform

}  // namespace usmp
}  // namespace tir
}  // namespace tvm
