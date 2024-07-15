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
 * \file tir/usmp/utils.cc
 * \brief Utilities for Unified Static Memory Planner
 */

#include <tvm/ir/memory_pools.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {

BufferInfo::BufferInfo(String name_hint, Integer size_bytes, Array<PoolInfo> pool_candidates,
                       Integer alignment, BufferInfoKind kind) {
  auto bufinfo_node = make_object<BufferInfoNode>();
  bufinfo_node->name_hint = name_hint;
  bufinfo_node->size_bytes = size_bytes;
  bufinfo_node->pool_candidates = pool_candidates;
  bufinfo_node->alignment = alignment;
  bufinfo_node->kind = kind;
  data_ = std::move(bufinfo_node);
}

void BufferInfoNode::SetConflicts(Array<ObjectRef> conflicting_buffer_info_objs) {
  this->conflicts = conflicting_buffer_info_objs;
}

TVM_REGISTER_NODE_TYPE(BufferInfoNode);
TVM_REGISTER_GLOBAL("tir.usmp.BufferInfo")
    .set_body_typed([](String name_hint, Integer size_bytes, Array<PoolInfo> pool_candidates,
                       Integer alignment) {
      if (!alignment.defined()) {
        return BufferInfo(name_hint, size_bytes, pool_candidates);
      }
      return BufferInfo(name_hint, size_bytes, pool_candidates, alignment);
    });
TVM_REGISTER_GLOBAL("tir.usmp.BufferInfoSetConflicts")
    .set_body_method<BufferInfo>(&BufferInfoNode::SetConflicts);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BufferInfoNode*>(ref.get());
      std::unordered_map<BufferInfoKind, String> toString = {
          {BufferInfoKind::kIntermediate, "kIntermediate"},
          {BufferInfoKind::kInput, "kInput"},
          {BufferInfoKind::kOutput, "kOutput"}};
      p->stream << "BufferInfoNode(\n"
                << "name_hint=" << node->name_hint << ",\n  size_bytes=" << node->size_bytes
                << ",\n  pool_candidates=" << node->pool_candidates
                << ",\n  alignment=" << node->alignment << ",\n  kind=" << toString[node->kind]
                << ",\n  conflicts=" << node->conflicts.size() << ")";
    });

BufferInfoAnalysis::BufferInfoAnalysis(Map<BufferInfo, tir::Stmt> buffer_info_stmts,
                                       Integer memory_pressure) {
  auto bufinfo_analysis_node = make_object<BufferInfoAnalysisNode>();
  bufinfo_analysis_node->buffer_info_stmts = buffer_info_stmts;
  bufinfo_analysis_node->memory_pressure = memory_pressure;
  data_ = std::move(bufinfo_analysis_node);
}

TVM_REGISTER_NODE_TYPE(BufferInfoAnalysisNode);
TVM_REGISTER_GLOBAL("tir.usmp.BufferInfoAnalysis")
    .set_body_typed([](Map<BufferInfo, tir::Stmt> buffer_info_stmts, Integer memory_pressure) {
      return BufferInfoAnalysis(buffer_info_stmts, memory_pressure);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferInfoAnalysisNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BufferInfoAnalysisNode*>(ref.get());
      p->stream << "BufferInfoAnalysisNode(\n"
                << "buffer_info_stmts=" << node->buffer_info_stmts
                << ",\n  memory_pressure=" << node->memory_pressure << ")";
    });

PoolAllocation::PoolAllocation(PoolInfo pool_info, Integer byte_offset) {
  auto pool_allocation_node = make_object<PoolAllocationNode>();
  pool_allocation_node->pool_info = pool_info;
  pool_allocation_node->byte_offset = byte_offset;
  data_ = std::move(pool_allocation_node);
}

TVM_REGISTER_NODE_TYPE(PoolAllocationNode);
TVM_REGISTER_GLOBAL("tir.usmp.PoolAllocation")
    .set_body_typed([](PoolInfo pool_info, Integer byte_offset) {
      return PoolAllocation(pool_info, byte_offset);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PoolAllocationNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PoolAllocationNode*>(ref.get());
      p->stream << "PoolAllocationNode(\n"
                << "pool_info=" << node->pool_info << ",\n  byte_offset=" << node->byte_offset
                << ")";
    });

AllocatedPoolInfo::AllocatedPoolInfo(PoolInfo pool_info, Integer allocated_size,
                                     Integer pool_var_idx) {
  auto allocated_poolinfo_node = make_object<AllocatedPoolInfoNode>();
  allocated_poolinfo_node->pool_info = pool_info;
  allocated_poolinfo_node->allocated_size = allocated_size;
  if (pool_var_idx.defined()) {
    allocated_poolinfo_node->pool_var_idx = pool_var_idx;
  }
  data_ = std::move(allocated_poolinfo_node);
}

TVM_REGISTER_NODE_TYPE(AllocatedPoolInfoNode);
TVM_REGISTER_GLOBAL("ir.AllocatedPoolInfo")
    .set_body_typed([](PoolInfo pool_info, Integer allocated_size, Integer pool_var_idx) {
      return AllocatedPoolInfo(pool_info, allocated_size, pool_var_idx);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AllocatedPoolInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const AllocatedPoolInfoNode*>(ref.get());
      p->stream << "AllocatedPoolInfoNode(\n"
                << "pool_info=" << node->pool_info << ",\n  allocated_size=" << node->allocated_size
                << ")";
    });

Array<BufferInfo> ConvertToArrayOfBufferInfo(const Map<BufferInfo, Stmt>& buffer_info_map) {
  Array<BufferInfo> ret;
  for (const auto& kv : buffer_info_map) {
    auto buffer_info = kv.first;
    ret.push_back(buffer_info);
  }
  return ret;
}

Map<Stmt, PoolAllocation> AssignStmtPoolAllocations(
    const Map<BufferInfo, Stmt>& buffer_info_to_stmt,
    const Map<BufferInfo, PoolAllocation>& buffer_info_to_pool_allocation) {
  Map<Stmt, PoolAllocation> ret;
  for (const auto& kv : buffer_info_to_pool_allocation) {
    BufferInfo bi = kv.first;
    Stmt stmt_ = buffer_info_to_stmt[bi];
    PoolAllocation pa = kv.second;
    ret.Set(stmt_, pa);
  }
  return ret;
}

Map<String, PoolAllocation> GetIOPoolAllocations(
    const Map<BufferInfo, PoolAllocation>& buffer_info_to_pool_allocation) {
  Map<String, PoolAllocation> io_tensor_name_to_pool_allocation;
  for (const auto& kv : buffer_info_to_pool_allocation) {
    BufferInfo buffer_info = kv.first;
    PoolAllocation pool_allocation = kv.second;
    if (buffer_info->kind != BufferInfoKind::kIntermediate) {
      io_tensor_name_to_pool_allocation.Set(buffer_info->name_hint, pool_allocation);
    }
  }
  return io_tensor_name_to_pool_allocation;
}

static Integer CalculateExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents) {
  if (dtype.is_scalable_vector()) {
    // We cannot statically calculate workspace for scalable types
    return Integer();
  }
  size_t element_size_bytes = dtype.bytes() * dtype.lanes();
  size_t num_elements = 1;
  for (const auto& ext : extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    } else {
      // We can't statically calculate workspace for dynamic shapes
      return Integer();
    }
  }
  return Integer(num_elements * element_size_bytes);
}

Integer CalculateExtentsSize(const AllocateNode* op) {
  return CalculateExtentsSize(op->dtype, op->extents);
}

Integer CalculateExtentsSize(const AllocateConstNode* op) {
  return CalculateExtentsSize(op->dtype, op->extents);
}

class ModuleWorkspaceSizeCalculator : public StmtExprVisitor {
 public:
  explicit ModuleWorkspaceSizeCalculator(const IRModule& module) : mod_(module) {
    for (const auto& gv_func : mod_->functions) {
      if ((gv_func.second)->IsInstance<tir::PrimFuncNode>()) {
        functions_.Set(gv_func.first->name_hint, Downcast<PrimFunc>(gv_func.second));
      }
    }
    main_func_ = Downcast<PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
    ICHECK(main_func_.defined()) << "main function is not in the module";
    Optional<Target> target_host = main_func_->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target_host) << "main function does not have a target attr";
    target_host_ = target_host.value();
  }

  Integer operator()() {
    UpdateWorkspaceData(main_func_);
    return Integer(max_workspace_size);
  }

 private:
  void UpdateWorkspaceData(const PrimFunc& func) {
    Target tgt = func->GetAttr<Target>(tvm::attr::kTarget).value_or(target_host_);
    Integer workspace_byte_alignment =
        tgt->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
    Integer workspace_req = CalculateWorkspaceBytes(func, workspace_byte_alignment);
    if (workspace_req.IntValue() != 0) {
      current_workspace_size_ += workspace_req->value;
    }
    if (max_workspace_size < current_workspace_size_) {
      max_workspace_size = current_workspace_size_;
    }
    this->VisitStmt(func->body);
    if (workspace_req.IntValue() != 0) {
      current_workspace_size_ -= workspace_req->value;
    }
  }

  void VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::call_extern())) {
      PrimFunc func = functions_.at(Downcast<StringImm>(op->args[0])->value);
      UpdateWorkspaceData(func);
    } else if (op->op->IsInstance<PrimFuncNode>()) {
      PrimFunc func = Downcast<PrimFunc>(op->op);
      UpdateWorkspaceData(func);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  IRModule mod_;
  Target target_host_;
  PrimFunc main_func_;
  Map<String, PrimFunc> functions_;
  size_t current_workspace_size_ = 0;
  size_t max_workspace_size = 0;
};

Integer CalculateModuleWorkspaceSize(const IRModule& mod) {
  return ModuleWorkspaceSizeCalculator(mod)();
}

TVM_REGISTER_GLOBAL("tir.usmp.CreateArrayBufferInfo")
    .set_body_typed([](Map<BufferInfo, Stmt> buffer_info_map) {
      return (ConvertToArrayOfBufferInfo(buffer_info_map));
    });

TVM_REGISTER_GLOBAL("tir.usmp.AssignStmtPoolAllocations").set_body_typed(AssignStmtPoolAllocations);

}  // namespace usmp
}  // namespace tir
}  // namespace tvm
