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
        Downcast<PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_run_func_suffix));
    ICHECK(main_func.defined()) << "main function is not in the module";
    Optional<Target> target_host = main_func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target_host) << "main function does not have a target attr";
    Array<usmp::PoolInfo> pool_infos =
        module->GetAttr<Array<usmp::PoolInfo>>(tvm::attr::kPoolInfoIRModuleAttr)
            .value_or({usmp::PoolInfo(
                "global_workspace", {{target_host.value(), PoolInfo::kTargetPoolReadWriteAccess}},
                PoolInfo::kUnrestrictedPoolSizeHint, PoolInfo::kUnknownClockFrequency,
                PoolInfo::kUnknownReadBandwidth, PoolInfo::kUnknownWriteBandwidth, 0, 0,
                {{target_host.value(), 1}}, Bool(true))});
    for (const usmp::PoolInfo& pool_info : pool_infos) {
      for (const auto& kv : pool_info->target_access) {
        Target tgt = kv.first;
        if (target_pool_infos_.find(tgt) == target_pool_infos_.end()) {
          target_pool_infos_.Set(tgt, Array<usmp::PoolInfo>());
        }
        Array<usmp::PoolInfo> pool_info_arr = target_pool_infos_[tgt];
        pool_info_arr.push_back(pool_info);
        target_pool_infos_.Set(tgt, pool_info_arr);
      }
    }
    mod_ = module->ShallowCopy();
  }

  IRModule operator()();

 private:
  Stmt VisitStmt_(const AllocateNode* op) override;

  IRModule mod_;
  Map<Target, Array<PoolInfo>> target_pool_infos_;
  PrimFunc func_;
};

Stmt PoolInfoAssigner::VisitStmt_(const AllocateNode* op) {
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following PrimFunc does not have a target attr: \n" << func_;
  Map<String, ObjectRef> annotations = Map<String, ObjectRef>(op->annotations);
  if (op->annotations.find(kPoolCandidatesAllocateAttr) == op->annotations.end()) {
    annotations.Set(kPoolCandidatesAllocateAttr, target_pool_infos_[tgt.value()]);
  }
  Stmt body = VisitStmt(op->body);
  auto allocate =
      Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body, annotations);
  return allocate;
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
