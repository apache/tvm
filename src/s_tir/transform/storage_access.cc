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
 * \file storage_access.cc
 */
#include "storage_access.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <string>
#include <utility>

#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallNode>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const TensorLoadNode* op) {
  Var buf = ResolveBuffer(op->source.as_or_throw<tvm::tirx::TensorVar>().var());
  StorageScope scope = StorageScope::Create(op->source.as_or_throw<tvm::tirx::TensorVar>().scope());
  if (Enabled(buf.get(), scope)) {
    TVM_FFI_ICHECK(allow_append_) << op << " " << scope.to_string();
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->ty.as_or_throw<PrimType>().WithLanes(1);
    for (const auto& index : op->indices) {
      e.touched.push_back(sym::IntSet::Vector(index));
    }
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  return StmtExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const TensorStoreNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;

  Var buf = ResolveBuffer(op->dest.as_or_throw<TensorVar>().var());
  StorageScope scope = StorageScope::Create(op->dest.as_or_throw<TensorVar>().scope());
  if (Enabled(buf.get(), scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->value.ty().WithLanes(1);
    for (const auto& index : op->indices) {
      e.touched.push_back(sym::IntSet::Vector(index));
    }
    e.type = kWrite;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const EvaluateNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
  // push to the scope
  if (curr_stmt_.access.size() != 0) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const BindNode* op) {
  if (const auto* call = op->value.as<CallNode>();
      call && call->op.same_as(tirx::decl_tensor_op())) {
    if (auto source = GetBufferDataVar(call->args[0])) {
      buffer_aliases_.insert_or_assign(op->var.as_or_throw<TensorVar>().get(),
                                       ResolveBuffer(source.value()));
    }
    return StmtExprVisitor::Visit_(op);
  }
  if (const auto* call = op->value.as<CallNode>();
      call && call->op.same_as(tirx::alloc_tensor_op()))
    return StmtExprVisitor::Visit_(op);
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const RegionStmtNode* op) {
  if (op->op.same_as(s_tir::manual_sync())) {
    // Trust the region's explicit barriers instead of planning access-based synchronization.
    return std::nullopt;
  }
  if (op->op.same_as(tirx::launch_thread_op()) &&
      std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0) {
    PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
    // IterVars are private access-analysis metadata, not launch definitions.
    env_threads_.push_back(IterVar(Range::FromMinExtent(IntImm(extent.ty(), 0), extent),
                                   op->body_params[0].as_or_throw<PrimVar>(),
                                   IterVarType::kThreadIndex,
                                   op->args[0].as_or_throw<StringImm>()->value));
    if (!in_device_env_) {
      in_device_env_ = true;
      scope_.push_back(std::vector<StmtEntry>());
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      // Separate kernels do not need an intervening thread barrier.
      Summarize(std::move(scope_.back()), nullptr);
      scope_.pop_back();
      in_device_env_ = false;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    env_threads_.pop_back();
    return std::nullopt;
  }
  // Preserve a distinct access scope for other region operations.
  scope_.push_back(std::vector<StmtEntry>());
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
  StmtEntry entry;
  entry.stmt = op;
  entry.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  if (!entry.access.empty()) scope_.back().push_back(std::move(entry));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const ForNode* op) {
  scope_.push_back(std::vector<StmtEntry>());
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), op);
  scope_.pop_back();
  if (s.access.size() != 0) {
    // relax the touched set to contain all ranges in the loop.
    std::unordered_map<const VarNode*, sym::IntSet> relax_map;
    relax_map[op->loop_var.get()] =
        sym::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
    for (AccessEntry& e : s.access) {
      if (e.buffer.defined()) {
        TVM_FFI_ICHECK(e.touched.size());
        ffi::Array<sym::IntSet> new_touched;
        for (const auto& touched : e.touched) {
          new_touched.push_back(sym::EvalSet(touched, relax_map));
        }
        e.touched = std::move(new_touched);
      }
    }
  }
  if (!s.access.empty()) {
    scope_.back().emplace_back(std::move(s));
  }
  return std::nullopt;
}

bool IsThreadInvariant(const PrimExpr& cond) {
  if (auto call = cond.as<CallNode>()) {
    if (auto opt_call_op = call->op.as<Op>()) {
      auto call_op = opt_call_op.value();
      if (call_op.same_as(tirx::gpu_thread_invariant_op())) {
        return true;
      }
    }
  }
  return false;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const IfNode* op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  scope_.push_back(std::vector<StmtEntry>());
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->then_case));
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  if (op->else_case) {
    scope_.push_back(std::vector<StmtEntry>());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->else_case.value()));
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const WhileNode* op) {
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  scope_.push_back(std::vector<StmtEntry>());
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->body));
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StorageAccessVisitor::Visit_(const CallNode* op) {
  Call call = ffi::GetRef<Call>(op);
  if (allow_append_ && in_opaque_call_) {
    ffi::Optional<TensorVar> buffer;
    if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      buffer = op->args[0].as<TensorVar>();
    } else if (op->op.same_as(tirx::address_of_op())) {
      if (const auto* load = op->args[0].as<TensorLoadNode>()) {
        buffer = load->source.as<TensorVar>();
      }
    }
    if (buffer.has_value()) {
      Var root = ResolveBuffer(buffer.value().var());
      StorageScope scope = GetScope(root);
      if (Enabled(root.get(), scope)) {
        AccessEntry entry;
        entry.threads = env_threads();
        entry.buffer = root;
        entry.dtype = buffer.value()->dtype;
        entry.scope = scope;
        // Without direction/extent metadata an escaping pointer may access any
        // element.  Retain conservative synchronization using existing entries.
        TensorVar storage =
            root->ty.as<TensorTypeNode>() ? root.as_or_throw<TensorVar>() : buffer.value();
        for (const PrimExpr& extent : storage->shape) {
          entry.touched.push_back(sym::IntSet::FromRange(Range::FromMinExtent(0, extent)));
        }
        entry.type = kRead;
        curr_stmt_.access.push_back(entry);
        entry.type = kWrite;
        curr_stmt_.access.push_back(std::move(entry));
      }
    }
  }
  if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
    bool is_load = op->op.same_as(tirx::masked_load_op());
    TensorVar buffer = op->args[0].as_or_throw<TensorVar>();
    PrimType value_dtype =
        is_load ? op->ty.as_or_throw<PrimType>() : op->args[1].as_or_throw<PrimExpr>().ty();
    Var buf = ResolveBuffer(buffer.var());
    StorageScope scope = StorageScope::Create(buffer.scope());
    if (Enabled(buf.get(), scope)) {
      TVM_FFI_ICHECK(allow_append_) << call << " " << scope.to_string();
      AccessEntry e;
      e.threads = env_threads();
      e.buffer = buf;
      e.dtype = value_dtype.WithLanes(1);
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        e.touched.push_back(sym::IntSet::Vector(op->args[i].as_or_throw<PrimExpr>()));
      }
      e.type = is_load ? kRead : kWrite;
      e.scope = scope;
      curr_stmt_.access.emplace_back(std::move(e));
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
  } else if (op->op.same_as(tirx::address_of_op())) {
    if (const auto* load = op->args[0].as<TensorLoadNode>()) {
      // Taking an address does not read the buffer value.  Visit only the
      // load's children so index expressions still contribute accesses.
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(load));
    } else {
      // address_of also accepts scalar variables (e.g. tcgen registers).
      // Recurse without assuming the argument is a TensorLoad.
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
  } else if (op->op.same_as(tirx::gpu_storage_sync_op())) {
    TVM_FFI_ICHECK(allow_append_);
    const std::string& s = op->args[0].as<StringImmNode>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::Create(s);
      AccessEntry e;
      e.threads = env_threads();
      e.type = kSync;
      e.scope = StorageScope::Create(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    bool previous = in_opaque_call_;
    auto effect_map = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    auto callee = op->op.as<Op>();
    if (!callee.has_value() || !effect_map.count(callee.value()) ||
        effect_map[callee.value()] > static_cast<int>(CallEffectKind::kPure)) {
      in_opaque_call_ = true;
    }
    auto result = StmtExprVisitor::Visit_(op);
    in_opaque_call_ = previous;
    return result;
  }
  return std::nullopt;
}

StorageScope StorageAccessVisitor::GetScope(Var buffer_var) const {
  if (auto buffer_type = buffer_var->ty.as<TensorType>()) {
    return StorageScope::Create(buffer_type.value()->storage_scope);
  }
  if (buffer_var->ty.as<PointerTypeNode>()) {
    return StorageScope::Create(GetPtrStorageScope(buffer_var));
  }
  return StorageScope();  // global by default
}

Var StorageAccessVisitor::ResolveBuffer(Var buffer_var) const {
  auto it = buffer_aliases_.find(buffer_var.get());
  return it == buffer_aliases_.end() ? buffer_var : it->second;
}

}  // namespace s_tir
}  // namespace tvm
