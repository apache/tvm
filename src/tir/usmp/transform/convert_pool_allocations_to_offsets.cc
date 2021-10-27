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

/*!
 * \file tir/analysis/usmp/transform/convert_pool_allocations_to_offsets.cc
 * \brief This pass would convert the pool allocations to offsets from pools
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>

namespace tvm {
namespace tir {
namespace usmp {

class PoolAllocationToOffsetConverter : public StmtExprMutator {
 public:
  explicit PoolAllocationToOffsetConverter(const IRModule& module,
                                           const Map<tir::Stmt, PoolAllocation>& pool_allocations)
      : pool_allocations_(pool_allocations) {
    module_ = module->ShallowCopy();
    for (const auto& gv_func : module_->functions) {
      function_global_vars_.Set(gv_func.first->name_hint, gv_func.first);
    }
    for (const auto& kv : pool_allocations) {
      // TODO(@manupa-arm): add AllocateConstNode when it is available
      ICHECK(kv.first->IsInstance<AllocateNode>());
      Allocate allocate_node = Downcast<Allocate>(kv.first);
      PoolAllocation pool_allocation = kv.second;
      PoolInfo pool_info = pool_allocation->pool_info;
      pool_ordering_.insert(pool_info);
      int byte_pool_offset = pool_allocation->byte_offset->value;
      int required_pool_size_for_allocation =
          byte_pool_offset + CalculateExtentsSize(allocate_node.operator->());
      if (all_pools_sizes_.find(pool_info) == all_pools_sizes_.end()) {
        all_pools_sizes_[pool_info] = required_pool_size_for_allocation;
      } else {
        int prev_required_pool_size = all_pools_sizes_[pool_info];
        if (prev_required_pool_size < required_pool_size_for_allocation) {
          all_pools_sizes_[pool_info] = required_pool_size_for_allocation;
        }
      }
    }
  }
  IRModule operator()();

 private:
  PrimExpr VisitExpr_(const CallNode* op) override;
  Stmt VisitStmt_(const AllocateNode* op) override;
  PrimExpr VisitExpr_(const LoadNode* op) override;
  Stmt VisitStmt_(const StoreNode* op) override;

  /*! \brief This is a structure where the modified function
   * signature is kept while body of the function is mutated
   */
  struct ScopeInfo {
    Array<tir::Var> params;
    Map<PoolInfo, tir::Var> pools_to_params;
    Map<tir::Var, Buffer> buffer_map;
  };

  /*! \brief The function scope information that are needed
   * in the mutation of the function need to be stacked and
   * popped when each function is entered/exited in the
   * mutation process.
   */
  std::stack<ScopeInfo> scope_stack;
  /*! \brief Each PrimFunc signature needs to be updated
   * with pool variables. This is a helper function to
   * capture the updated information to ScopeInfo object.
   */
  ScopeInfo UpdateFunctionScopeInfo(const PrimFunc& original_func);
  /*! \brief This is a helper to create the PrimFunc with
   * pool variables that calls the UpdateFunctionScopeInfo
   * inside of it.
   */
  PrimFunc CreatePrimFuncWithPoolParams(const PrimFunc& original_primfunc);
  /*! \brief This is a helper to append the pool args to
   * the callsite of the function.
   */
  Array<PrimExpr> AppendPoolParamsToArgs(const CallNode* op);
  /*! \brief Some arguments that used to be Allocate nodes
   * should be replaced by Let nodes in the pass that loads
   * the space from a pool variable.
   */
  Array<PrimExpr> ReplaceAllocateArgsWithLetArgs(const Array<PrimExpr>& args);

  /*! \brief The tir::Var map to PoolInfo objects */
  Map<tir::Var, PoolInfo> primfunc_args_to_pool_info_map_;
  /*! \brief The buffer var map to their allocate nodes */
  Map<tir::Var, tir::Stmt> allocate_var_to_stmt_map_;
  /*! \brief The IRModule being constructed/mutated */
  IRModule module_;
  /*! \brief The input allocate node to PoolAllocation map */
  Map<tir::Stmt, PoolAllocation> pool_allocations_;
  /*! \brief The set of ordered pools to ensure an unique order of args for functions */
  std::set<PoolInfo> pool_ordering_;
  /*! \brief The storage of calculated pool size at init */
  std::unordered_map<PoolInfo, int, ObjectPtrHash, ObjectPtrEqual> all_pools_sizes_;
  /*! \brief The AoT codegen uses extern_calls due to some functions not being exposed in the TIR
   * IRModule This maps maintains the map of which to each function
   */
  Map<String, GlobalVar> function_global_vars_;
  /*! \brief After mutation, each allocate buffer is replaced with tir::Var that is let bounded
   * to position from a pool as designated by a PoolAllocation
   */
  Map<tir::Var, tir::Var> allocate_buf_to_let_var_;
  /*! \brief A counter to give references to pools a reproducible unique set of names */
  int pool_var_count_ = 0;
};

PoolAllocationToOffsetConverter::ScopeInfo PoolAllocationToOffsetConverter::UpdateFunctionScopeInfo(
    const PrimFunc& original_func) {
  ScopeInfo si;
  si.params = original_func->params;
  si.buffer_map = original_func->buffer_map;
  Map<tir::Var, PoolInfo> ret;
  for (const PoolInfo& pool_info : pool_ordering_) {
    String pool_ref_name = pool_info->pool_name + "_" + std::to_string(pool_var_count_++);
    String var_name = pool_ref_name + "_var";
    DataType elem_dtype = DataType::UInt(8);
    Var buffer_var(var_name, PointerType(PrimType(elem_dtype), "global"));
    Var pool_var(var_name, DataType::Handle());
    si.params.push_back(pool_var);
    si.pools_to_params.Set(pool_info, pool_var);

    int pool_size = all_pools_sizes_[pool_info];
    String buffer_var_name = pool_ref_name + "_buffer_var";
    si.buffer_map.Set(pool_var, Buffer(buffer_var, elem_dtype, {pool_size}, {1}, 1, buffer_var_name,
                                       16, 1, BufferType::kDefault));
  }
  return si;
}

PrimFunc PoolAllocationToOffsetConverter::CreatePrimFuncWithPoolParams(
    const PrimFunc& original_primfunc) {
  ScopeInfo si = UpdateFunctionScopeInfo(original_primfunc);
  this->scope_stack.push(si);
  Stmt new_body = this->VisitStmt(original_primfunc->body);
  this->scope_stack.pop();
  return PrimFunc(si.params, new_body, original_primfunc->ret_type, si.buffer_map,
                  original_primfunc->attrs);
}

Array<PrimExpr> PoolAllocationToOffsetConverter::AppendPoolParamsToArgs(const CallNode* op) {
  Array<PrimExpr> new_args;
  for (const auto& arg : op->args) {
    new_args.push_back(VisitExpr(arg));
  }
  for (const auto& pools_vars : this->scope_stack.top().pools_to_params) {
    tir::Var pool_var = pools_vars.second;
    new_args.push_back(pool_var);
  }
  return new_args;
}

Array<PrimExpr> PoolAllocationToOffsetConverter::ReplaceAllocateArgsWithLetArgs(
    const Array<PrimExpr>& args) {
  Array<PrimExpr> ret;
  for (const PrimExpr& arg : args) {
    if (arg->IsInstance<VarNode>() &&
        allocate_buf_to_let_var_.find(Downcast<Var>(arg)) != allocate_buf_to_let_var_.end()) {
      ret.push_back(allocate_buf_to_let_var_[Downcast<Var>(arg)]);
    } else {
      ret.push_back(arg);
    }
  }
  return ret;
}

PrimExpr PoolAllocationToOffsetConverter::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern())) {
    String func_name = Downcast<StringImm>(op->args[0])->value;
    GlobalVar gv = function_global_vars_.at(func_name);
    PrimFunc func = Downcast<PrimFunc>(module_->Lookup(gv));
    PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
    module_->Update(gv, prim_func);
    Array<PrimExpr> new_args = AppendPoolParamsToArgs(op);
    new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    return Call(op->dtype, builtin::call_extern(), new_args);
  } else if (op->op->IsInstance<PrimFuncNode>()) {
    PrimFunc func = Downcast<PrimFunc>(op->op);
    PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
    Array<PrimExpr> new_args = AppendPoolParamsToArgs(op);
    new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    return Call(op->dtype, prim_func, new_args);
  } else {
    return StmtExprMutator::VisitExpr_(op);
  }
}

Stmt PoolAllocationToOffsetConverter::VisitStmt_(const AllocateNode* op) {
  if (pool_allocations_.count(GetRef<Allocate>(op))) {
    ScopeInfo scope_info = scope_stack.top();
    PoolAllocation pool_allocation = pool_allocations_[GetRef<Allocate>(op)];
    Var param = scope_info.pools_to_params[pool_allocation->pool_info];
    Buffer buffer_var = scope_info.buffer_map[param];
    ICHECK(pool_allocation->byte_offset < all_pools_sizes_[pool_allocation->pool_info]);
    Load load_node = Load(op->dtype, buffer_var->data, pool_allocation->byte_offset, op->condition);
    Var tir_var(op->buffer_var->name_hint + "_let", op->dtype);
    allocate_buf_to_let_var_.Set(op->buffer_var, tir_var);
    Stmt new_body = VisitStmt(op->body);
    allocate_buf_to_let_var_.erase(op->buffer_var);
    return LetStmt(tir_var, load_node, new_body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt PoolAllocationToOffsetConverter::VisitStmt_(const StoreNode* op) {
  if (allocate_buf_to_let_var_.find(op->buffer_var) != allocate_buf_to_let_var_.end()) {
    return Store(allocate_buf_to_let_var_[op->buffer_var], VisitExpr(op->value), op->index,
                 VisitExpr(op->predicate));
  }
  return StmtExprMutator::VisitStmt_(op);
}

PrimExpr PoolAllocationToOffsetConverter::VisitExpr_(const LoadNode* op) {
  if (allocate_buf_to_let_var_.find(op->buffer_var) != allocate_buf_to_let_var_.end()) {
    return Load(op->dtype, allocate_buf_to_let_var_[op->buffer_var], op->index,
                VisitExpr(op->predicate));
  }
  return StmtExprMutator::VisitExpr_(op);
}

IRModule PoolAllocationToOffsetConverter::operator()() {
  GlobalVar gv = function_global_vars_.at(::tvm::runtime::symbol::tvm_run_func_suffix);
  PrimFunc main_func = Downcast<PrimFunc>(module_->Lookup(gv));
  ScopeInfo si = UpdateFunctionScopeInfo(main_func);
  this->scope_stack.push(si);
  Stmt main_func_body = this->VisitStmt(main_func->body);
  this->scope_stack.pop();
  module_->Update(gv, PrimFunc(si.params, main_func_body, main_func->ret_type, si.buffer_map,
                               main_func->attrs));
  return this->module_;
}

namespace transform {

tvm::transform::Pass ConvertPoolAllocationsToOffsets(
    const Map<tir::Stmt, PoolAllocation>& pool_allocations) {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return Downcast<IRModule>(PoolAllocationToOffsetConverter(m, pool_allocations)());
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.usmp.ConvertPoolAllocationsToOffsets",
                                          {});
}

TVM_REGISTER_GLOBAL("tir.usmp.transform.ConvertPoolAllocationsToOffsets")
    .set_body_typed(ConvertPoolAllocationsToOffsets);

}  // namespace transform

}  // namespace usmp
}  // namespace tir
}  // namespace tvm
