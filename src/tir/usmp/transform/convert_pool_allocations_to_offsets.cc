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
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>

namespace tvm {
namespace tir {
namespace usmp {

/*!
 * \brief The StmtExpr mutator class to replace allocate nodes
 * with offsets within memory pools
 *
 * This mutator class will add Pool variables recursively to every PrimFunc
 * starting from the main PrimFunc. For all allocate nodes, that have been
 * memory planned, will be mutated into an offset using a Let binding.
 */
class PoolAllocationToOffsetConverter : public StmtExprMutator {
 public:
  PoolAllocationToOffsetConverter(const IRModule& module,
                                  const Map<tir::Stmt, PoolAllocation>& pool_allocations,
                                  bool emit_tvmscript_printable = false)
      : pool_allocations_(pool_allocations), emit_tvmscript_printable_(emit_tvmscript_printable) {
    module_ = module->ShallowCopy();
    for (const auto& kv : pool_allocations) {
      // TODO(@manupa-arm): add AllocateConstNode when it is available
      ICHECK(kv.first->IsInstance<AllocateNode>());
      Allocate allocate_node = Downcast<Allocate>(kv.first);
      PoolAllocation pool_allocation = kv.second;
      PoolInfo pool_info = pool_allocation->pool_info;
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

    for (const auto& kv : all_pools_sizes_) {
      PoolInfo pi = kv.first;
      int allocated_size = kv.second;
      allocated_pool_ordering_.push_back(AllocatedPoolInfo(pi, allocated_size));
    }
    std::sort(allocated_pool_ordering_.begin(), allocated_pool_ordering_.end(),
              [](const AllocatedPoolInfo& lhs, const AllocatedPoolInfo& rhs) {
                if (lhs->pool_info->pool_name < rhs->pool_info->pool_name) {
                  return true;
                }
                return false;
              });
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
    Array<AllocatedPoolInfo> allocated_pool_params;
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
  Array<PrimExpr> AppendPoolParamsToArgs(Array<PrimExpr> args, const PrimFunc& func);
  /*! \brief Some arguments that used to be Allocate nodes
   * should be replaced by Let nodes in the pass that loads
   * the space from a pool variable.
   */
  Array<PrimExpr> ReplaceAllocateArgsWithLetArgs(const Array<PrimExpr>& args);
  /*! \brief Obtain a resource handle if its there
   */
  Optional<Var> GetResourceHandle(const PrimFunc& func);

  /*! \brief The tir::Var map to PoolInfo objects */
  Map<tir::Var, PoolInfo> primfunc_args_to_pool_info_map_;
  /*! \brief The buffer var map to their allocate nodes */
  Map<tir::Var, tir::Stmt> allocate_var_to_stmt_map_;
  /*! \brief The IRModule being constructed/mutated */
  IRModule module_;
  /*! \brief The input allocate node to PoolAllocation map */
  Map<tir::Stmt, PoolAllocation> pool_allocations_;
  /*! \brief The set of ordered pools to ensure an unique order of args for functions */
  std::vector<AllocatedPoolInfo> allocated_pool_ordering_;
  /*! \brief The storage of calculated pool size at init */
  std::unordered_map<PoolInfo, int, ObjectPtrHash, ObjectPtrEqual> all_pools_sizes_;
  /*! \brief After mutation, each allocate buffer is replaced with tir::Var that is let bounded
   * to position from a pool as designated by a PoolAllocation
   */
  Map<tir::Var, tir::Var> allocate_buf_to_let_var_;
  /*! \brief A counter to give references to pools a reproducible unique set of names */
  int pool_var_count_ = 0;
  /*! \brief This toggles to remove non tvmscript printable items for IRModule for unit tests */
  bool emit_tvmscript_printable_ = false;
  /*! \brief A counter to give references to pools a reproducible unique set of names */
  std::unordered_set<PrimFunc, ObjectPtrHash, ObjectPtrEqual> visited_primfuncs;
};

Optional<Var> PoolAllocationToOffsetConverter::GetResourceHandle(const PrimFunc& func) {
  if (func->buffer_map.find(func->params.back()) == func->buffer_map.end()) {
    return func->params.back();
  }
  return Optional<Var>();
}

PoolAllocationToOffsetConverter::ScopeInfo PoolAllocationToOffsetConverter::UpdateFunctionScopeInfo(
    const PrimFunc& original_func) {
  ScopeInfo si;

  Optional<Var> resource_handle = GetResourceHandle(original_func);
  si.params = original_func->params;
  if (resource_handle) {
    si.params.pop_back();
    ICHECK(si.params.size() == original_func->params.size() - 1);
  }
  si.buffer_map = original_func->buffer_map;
  Map<tir::Var, PoolInfo> ret;
  for (const AllocatedPoolInfo& allocated_pool_info : allocated_pool_ordering_) {
    PoolInfo pool_info = allocated_pool_info->pool_info;
    String pool_ref_name = pool_info->pool_name + "_" + std::to_string(pool_var_count_++);
    String var_name = pool_ref_name + "_var";
    DataType elem_dtype = DataType::UInt(8);
    Var buffer_var(var_name, PointerType(PrimType(elem_dtype), "global"));
    Var pool_var;
    if (!emit_tvmscript_printable_) {
      pool_var = Var(var_name, PointerType(PrimType(elem_dtype), "global"));
    } else {
      pool_var = Var(var_name, DataType::Handle(8));
    }
    si.params.push_back(pool_var);
    si.pools_to_params.Set(pool_info, pool_var);
    si.allocated_pool_params.push_back(AllocatedPoolInfo(
        allocated_pool_info->pool_info, allocated_pool_info->allocated_size, si.params.size() - 1));

    int pool_size = all_pools_sizes_[pool_info];
    String buffer_var_name = pool_ref_name + "_buffer_var";
    si.buffer_map.Set(pool_var, Buffer(buffer_var, elem_dtype, {pool_size}, {1}, 1, buffer_var_name,
                                       16, 1, BufferType::kDefault));
  }
  if (resource_handle) {
    si.params.push_back(resource_handle.value());
  }
  return si;
}

PrimFunc PoolAllocationToOffsetConverter::CreatePrimFuncWithPoolParams(
    const PrimFunc& original_primfunc) {
  // Only create the new function if it was not modified with pool params
  if (visited_primfuncs.find(original_primfunc) == visited_primfuncs.end()) {
    ScopeInfo si = UpdateFunctionScopeInfo(original_primfunc);
    this->scope_stack.push(si);
    Stmt new_body = this->VisitStmt(original_primfunc->body);
    this->scope_stack.pop();
    DictAttrs original_attrs = original_primfunc->attrs;
    // We dont need attrs of PrimFunc that might include non printable attrs such as target
    // for unit tests where emit_tvmscript_printable_ is to be used.
    if (emit_tvmscript_printable_) {
      original_attrs = DictAttrs();
    }
    PrimFunc ret =
        PrimFunc(si.params, new_body, original_primfunc->ret_type, si.buffer_map, original_attrs);
    if (!emit_tvmscript_printable_) {
      ret = WithAttr(ret, tvm::attr::kPoolArgs, si.allocated_pool_params);
    }
    visited_primfuncs.insert(ret);
    return ret;
  }
  return original_primfunc;
}

Array<PrimExpr> PoolAllocationToOffsetConverter::AppendPoolParamsToArgs(Array<PrimExpr> args,
                                                                        const PrimFunc& func) {
  Array<PrimExpr> new_args;
  PrimExpr resource_handle_arg;
  if (args.size() == func->params.size() + 1) {
    resource_handle_arg = args.back();
    args.pop_back();
  }
  for (const auto& arg : args) {
    new_args.push_back(VisitExpr(arg));
  }
  ScopeInfo top_scope = this->scope_stack.top();
  for (const auto& pools_vars : top_scope.pools_to_params) {
    tir::Var pool_var = pools_vars.second;
    Buffer buffer_var = top_scope.buffer_map[pool_var];
    new_args.push_back(buffer_var->data);
  }
  if (resource_handle_arg.defined()) {
    new_args.push_back(resource_handle_arg);
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
      ret.push_back(VisitExpr(arg));
    }
  }
  return ret;
}

PrimExpr PoolAllocationToOffsetConverter::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
    String func_name = Downcast<StringImm>(op->args[0])->value;
    Array<PrimExpr> new_args;
    if (module_->ContainGlobalVar(func_name) &&
        module_->Lookup(func_name)->IsInstance<PrimFuncNode>()) {
      GlobalVar gv = module_->GetGlobalVar(func_name);
      PrimFunc func = Downcast<PrimFunc>(module_->Lookup(gv));
      PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
      module_->Update(gv, prim_func);
      new_args = AppendPoolParamsToArgs(op->args, prim_func);
      new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    } else {
      new_args = ReplaceAllocateArgsWithLetArgs(op->args);
    }
    return Call(op->dtype, op->op, new_args);
  }
  if (op->op->IsInstance<PrimFuncNode>()) {
    PrimFunc func = Downcast<PrimFunc>(op->op);
    PrimFunc prim_func = CreatePrimFuncWithPoolParams(func);
    Array<PrimExpr> new_args = AppendPoolParamsToArgs(op->args, prim_func);
    new_args = ReplaceAllocateArgsWithLetArgs(new_args);
    return Call(op->dtype, prim_func, new_args);
  }
  return StmtExprMutator::VisitExpr_(op);
}

Stmt PoolAllocationToOffsetConverter::VisitStmt_(const AllocateNode* op) {
  if (pool_allocations_.count(GetRef<Allocate>(op))) {
    ScopeInfo scope_info = scope_stack.top();
    PoolAllocation pool_allocation = pool_allocations_[GetRef<Allocate>(op)];
    Var param = scope_info.pools_to_params[pool_allocation->pool_info];
    Buffer buffer_var = scope_info.buffer_map[param];
    Load load_node =
        Load(DataType::UInt(8), buffer_var->data, pool_allocation->byte_offset, op->condition);
    Call address_of_load = Call(DataType::Handle(8), builtin::address_of(), {load_node});
    Var tir_var;
    if (!emit_tvmscript_printable_) {
      tir_var = Var(op->buffer_var->name_hint + "_let", op->buffer_var->type_annotation);
    } else {
      tir_var = Var(op->buffer_var->name_hint + "_let", DataType::Handle(8));
    }
    allocate_buf_to_let_var_.Set(op->buffer_var, tir_var);
    Stmt new_body = VisitStmt(op->body);
    allocate_buf_to_let_var_.erase(op->buffer_var);
    return LetStmt(tir_var, address_of_load, new_body);
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
  GlobalVar gv = module_->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main);
  PrimFunc main_func = Downcast<PrimFunc>(module_->Lookup(gv));
  ScopeInfo si = UpdateFunctionScopeInfo(main_func);
  this->scope_stack.push(si);
  Stmt main_func_body = this->VisitStmt(main_func->body);
  this->scope_stack.pop();
  // We dont need attrs of PrimFunc that might include non printable attrs such as target
  // for unit tests where emit_tvmscript_printable_ is to be used.
  if (!emit_tvmscript_printable_) {
    main_func =
        PrimFunc(si.params, main_func_body, main_func->ret_type, si.buffer_map, main_func->attrs);
    main_func = WithAttr(main_func, tvm::attr::kPoolArgs, si.allocated_pool_params);
  } else {
    main_func =
        PrimFunc(si.params, main_func_body, main_func->ret_type, si.buffer_map, DictAttrs());
  }
  module_->Update(gv, main_func);
  if (!emit_tvmscript_printable_) {
    return WithAttr(this->module_, tvm::attr::kPoolArgs, si.allocated_pool_params);
  }
  return this->module_;
}

namespace transform {

tvm::transform::Pass ConvertPoolAllocationsToOffsets(
    const Map<tir::Stmt, PoolAllocation>& pool_allocations, Bool emit_tvmscript_printable) {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return Downcast<IRModule>(PoolAllocationToOffsetConverter(
        m, pool_allocations, emit_tvmscript_printable->value != 0)());
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
