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
 * \file tir/analysis/apply_device_constraints.cc
 * \brief Applies device-related constraints to \p PrimFunc parameters.
 *
 * This is used by the \p PlanDevices pass to flow device-constraints *into* \p PrimFuncs.
 *
 * Currently only applies memory scope constraints into \p Buffer data pointer
 * storage scopes. Aliased ('matched') buffers take on any scope introduced on
 * the buffer they alias. However currently does not attempt to flow constraints into
 * allocated buffers.
 */

#include "./device_constraint_utils.h"

#include <tvm/relay/attrs/memory.h>
#include <tvm/target/virtual_device.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {
namespace {

/*!
 * \brief Returns the \p PointerTypeNode for \p buffer, or nullptr if \p buffer does not describe a
 * pointer.
 */
const PointerTypeNode* PointerInBuffer(const tir::Buffer& buffer) {
  return buffer->data->type_annotation.defined()
             ? buffer->data->type_annotation.as<PointerTypeNode>()
             : nullptr;
}

/*!
 * \brief Returns the parameter variable and corresponding buffer at or after \p
 * *current_primfunc_param_index in \p prim_func. Will skip over any non-pointer parameters. This
 * can be used to find the parameter matching a tensor type in a flattened Relay function parameter
 * or result.
 */
std::pair<tir::Var, tir::Buffer> FindPointerParam(const tir::PrimFunc& prim_func,
                                                  size_t* current_primfunc_param_index) {
  while (true) {
    ICHECK_LT(*current_primfunc_param_index, prim_func->params.size());
    const tir::Var& param = prim_func->params[*current_primfunc_param_index];
    auto itr = prim_func->buffer_map.find(param);
    if (itr == prim_func->buffer_map.end()) {
      VLOG(2) << "no buffer map entry for '" << param->name_hint << "'";
      ++*current_primfunc_param_index;
      continue;
    }
    const auto* pointer_type_node = PointerInBuffer((*itr).second);
    if (pointer_type_node == nullptr) {
      VLOG(2) << "not a pointer type for '" << param->name_hint << "'";
      ++*current_primfunc_param_index;
      continue;
    }
    VLOG(2) << "using PrimFunc param '" << param->name_hint << "'";
    return *itr;
  }
}

/*!
 * \brief Check fails if any parameter at or after \p *current_primfunc_param_index in \p prim_func
 * is for a pointer type. This can be used to check all \p prim_func parameters have been accounted
 * for when using \p FindPointerParam above.
 */
void CheckNoRemainingPointerParams(const tir::PrimFunc& prim_func,
                                   size_t* current_primfunc_param_index) {
  while (*current_primfunc_param_index < prim_func->params.size()) {
    const tir::Var& param = prim_func->params[*current_primfunc_param_index];
    auto itr = prim_func->buffer_map.find(param);
    if (itr == prim_func->buffer_map.end()) {
      VLOG(1) << "no buffer map entry for '" << param->name_hint << "'";
      ++*current_primfunc_param_index;
      continue;
    }
    const auto* pointer_type_node = PointerInBuffer((*itr).second);
    ICHECK(pointer_type_node == nullptr);
    ++*current_primfunc_param_index;
  }
}

/*!
 * \brief Returns the (consistent) constraint to use for a Relay parameter of \p type,
 * using \p prim_func parameters at or after \p *current_primfunc_param_index. Currently
 * only memory scope is extracted. Fails if constraints are not consistent, ie \p type is a tuple
 * type and the \p prim_func is attempting to map different fields of that tuple to different memory
 * scopes. Returns the fully unconstrained \p VirtualDevice if no memory scopes constraints arise
 * from the \p prim_func, ie all storage scope strings in pointer types are empty.
 */
VirtualDevice ConsistentParamConstraint(const tir::PrimFunc& prim_func, const Type& type,
                                        size_t* current_primfunc_param_index) {
  std::string memory_scope;  // default empty => no constraint
  for (size_t i = 0; i < relay::FlattenTupleType(type).size(); ++i) {
    std::pair<tir::Var, tir::Buffer> kv = FindPointerParam(prim_func, current_primfunc_param_index);
    const tir::Buffer& buffer = kv.second;
    const auto* pointer_type_node = buffer->data->type_annotation.as<PointerTypeNode>();
    const MemoryScope& buffer_memory_scope = pointer_type_node->storage_scope;
    if (memory_scope.empty()) {
      memory_scope = buffer_memory_scope;
    } else if (buffer_memory_scope.empty()) {
      // No constraint.
    } else {
      // Tuples must be homogenous on their VirtualDevice and thus memory scope.
      ICHECK_EQ(buffer_memory_scope, memory_scope);
    }
    ++*current_primfunc_param_index;
  }
  return VirtualDevice::ForMemoryScope(memory_scope);
}

/*!
 * \brief Insert into param_constraints an entry for each parameter of \p prim_func starting from
 * \p *current_primfunc_param_index for the flattened form of a Rleay parameters of \p type. Each
 * entry maps to \p virtual_device.
 */
void InsertParamConstraints(
    const tir::PrimFunc& prim_func, const Type& type, const VirtualDevice& virtual_device,
    size_t* current_primfunc_param_index,
    std::unordered_map<const tir::VarNode*, VirtualDevice>* param_constraints) {
  for (size_t i = 0; i < relay::FlattenTupleType(type).size(); ++i) {
    std::pair<tir::Var, tir::Buffer> kv = FindPointerParam(prim_func, current_primfunc_param_index);
    param_constraints->emplace(kv.first.get(), virtual_device);
    ++*current_primfunc_param_index;
  }
}

/*!
 * \brief Apply the memory scope constraints to the \p Buffers and data \p Vars of a \p PrimFunc.
 *
 * All definitional occurrences of buffer Vars are rewritten to capture memory scopes in their
 * PointerTypes:
 *  - Buffer::data (if the buffer itself is a definitional occurrence)
 *  - AllocateNode::buffer_var
 *  - FUTURE: LetStmtNode::var if aliasing a buffer data var.
 *
 * All referential occurrences of buffer Vars are replaced with their new definitions:
 *  - LoadNode::buffer_var
 *  - StoreNode::buffer_var
 *
 * Similarly all definitional occurrences of Buffers are rewritten to account for any new memory
 * scopes:
 *  - PrimFuncNode::buffer_map keys.
 *  - BlockNode::match_buffers.buffer
 *  - FUTURE: BlockNode::alloc_buffers?
 *
 * And all referential occurrences of Buffers are replaced with their new definitions:
 *  - BufferLoadNode::buffer
 *  - BufferStoreNode::buffer
 *  - BufferRealizeNode::buffer
 *  - PrefetchNode::buffer
 *  - BufferRegionNode:buffer
 *  - BlockNode.match_buffers.source.buffer
 *  - BlockNode::{reads, writes}.buffer
 *
 * CAUTION: We assume strict sharing of Buffer objects and do not attempt to rewrite the bodies
 * of referential buffers.
 *
 * CAUTION: EXPERIMENTAL: We don't yet account for all buffers and pointer types.
 */
class ApplyDeviceConstraintsMutator : public StmtExprMutator {
 public:
  ApplyDeviceConstraintsMutator() = default;

  /*!
   * \brief Returns \p prim_func written to capture the memory scope constraints in \p
   * param_constraints for each pointer \p prim_func parameter. Returns \p prim_func unchanged if no
   * memory scopes needed to change.
   */
  PrimFunc Rewrite(const PrimFunc& prim_func, const FuncType& relay_func_type,
                   const Array<VirtualDevice>& arg_and_result_virtual_devices) {
    size_t current_primfunc_param_index = 0;
    std::unordered_map<const tir::VarNode*, VirtualDevice> param_constraints;

    // For each Relay function parameter...
    for (size_t i = 0; i < relay_func_type->arg_types.size(); ++i) {
      const Type& param_type = relay_func_type->arg_types[i];
      const VirtualDevice& param_virtual_device = arg_and_result_virtual_devices[i];
      InsertParamConstraints(prim_func, param_type, param_virtual_device,
                             &current_primfunc_param_index, &param_constraints);
    }

    // For the Relay function result...
    const Type& ret_type = relay_func_type->ret_type;
    const VirtualDevice& ret_virtual_device = arg_and_result_virtual_devices.back();
    InsertParamConstraints(prim_func, ret_type, ret_virtual_device, &current_primfunc_param_index,
                           &param_constraints);

    // Make sure we accounted for all prim_func parameters.
    CheckNoRemainingPointerParams(prim_func, &current_primfunc_param_index);

    // Start with a copy of the current prim_func buffer map.
    Map<Var, Buffer> new_buffer_map(prim_func->buffer_map.begin(), prim_func->buffer_map.end());
    bool any_change = false;

    // For each constrained parameter...
    for (const auto& kv : param_constraints) {
      const tir::Var param = GetRef<tir::Var>(kv.first);
      const VirtualDevice& virtual_device = kv.second;
      const tir::Buffer& buffer = prim_func->buffer_map[param];
      // Rewrite the buffer to account for constraint.
      const Buffer new_buffer = RewriteBuffer(buffer, virtual_device);
      if (!new_buffer.same_as(buffer)) {
        any_change = true;
      }
      new_buffer_map.Set(param, new_buffer);
    }
    // Make sure we have accounted for all prim_func parameters.
    CheckNoRemainingPointerParams(prim_func, &current_primfunc_param_index);

    // Apply data variable and buffer substitutions to the prim_func body. These will have been
    // accumulated from processing the parameters above.
    Stmt new_body = VisitStmt(prim_func->body);
    if (!new_body.same_as(prim_func->body)) {
      any_change = true;
    }

    // We are done with the substitutions.
    var_subst_.clear();
    buffer_subst_.clear();

    if (any_change) {
      return PrimFunc(prim_func->params, std::move(new_body), prim_func->ret_type,
                      std::move(new_buffer_map), prim_func->attrs, prim_func->span);
    } else {
      return prim_func;
    }
  }

 private:
  PrimExpr VisitExpr_(const VarNode* var_node) final { return Subst(var_node); }

  PrimExpr VisitExpr_(const LoadNode* load_node) final {
    Load new_load = Downcast<Load>(StmtExprMutator::VisitExpr_(load_node));
    Var new_buffer_var = Subst(new_load->buffer_var.get());
    if (!new_buffer_var.same_as(new_load->buffer_var)) {
      return Load(load_node->dtype, new_buffer_var, load_node->index, load_node->predicate);
    }
    return std::move(new_load);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* buffer_load_node) final {
    BufferLoad new_buffer_load =
        Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(buffer_load_node));
    Buffer new_buffer = Subst(new_buffer_load->buffer.get());
    if (!new_buffer.same_as(new_buffer_load->buffer)) {
      return BufferLoad(new_buffer, new_buffer_load->indices, new_buffer_load->span);
    }
    return std::move(new_buffer_load);
  }

  Stmt VisitStmt_(const LetStmtNode* let_stmt_node) final {
    // TODO(mbs): If the let-bound var is aliasing an existing buffer data var we need to
    // rewrite it.
    return StmtExprMutator::VisitStmt_(let_stmt_node);
  }

  Stmt VisitStmt_(const AttrStmtNode* attr_stmt_node) final {
    AttrStmt new_attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(attr_stmt_node));
    // remap node if a var
    if (const auto* var_node = new_attr_stmt->node.as<VarNode>()) {
      Var new_var = Subst(var_node);
      if (!new_var.same_as(new_attr_stmt->node)) {
        return AttrStmt(new_var, new_attr_stmt->attr_key, new_attr_stmt->value,
                        new_attr_stmt->body);
      }
    }
    return std::move(new_attr_stmt);
  }

  // ForNode default ok since loop_var never of PointerType

  // WhileNode default ok

  Stmt VisitStmt_(const AllocateNode* allocate_node) final {
    // TODO(mbs): What memory scope should we assign to the new pointer?
    return StmtExprMutator::VisitStmt_(allocate_node);
  }

  Stmt VisitStmt_(const StoreNode* store_node) final {
    Store new_store = Downcast<Store>(StmtExprMutator::VisitStmt_(store_node));
    Var new_buffer_var = Subst(new_store->buffer_var.get());
    if (!new_buffer_var.same_as(new_store->buffer_var)) {
      Store(new_buffer_var, new_store->value, new_store->index, new_store->predicate);
    }
    return std::move(new_store);
  }

  Stmt VisitStmt_(const BufferStoreNode* buffer_store_node) final {
    BufferStore new_buffer_store =
        Downcast<BufferStore>(StmtExprMutator::VisitStmt_(buffer_store_node));
    Buffer new_buffer = Subst(new_buffer_store->buffer.get());
    if (!new_buffer.same_as(new_buffer_store->buffer)) {
      return BufferStore(new_buffer, new_buffer_store->value, new_buffer_store->indices,
                         new_buffer_store->span);
    }
    return std::move(new_buffer_store);
  }

  Stmt VisitStmt_(const BufferRealizeNode* buffer_realize_node) final {
    BufferRealize new_buffer_realize =
        Downcast<BufferRealize>(StmtExprMutator::VisitStmt_(buffer_realize_node));
    Buffer new_buffer = Subst(new_buffer_realize->buffer.get());
    if (!new_buffer.same_as(new_buffer_realize->buffer)) {
      return BufferRealize(new_buffer, new_buffer_realize->bounds, new_buffer_realize->condition,
                           new_buffer_realize->body, new_buffer_realize->span);
    }
    return std::move(new_buffer_realize);
  }

  // IfThenElseNode default ok
  // AssertStmtNode default ok
  // ProducerStoreNode default ok (though does not visit producer)
  // ProducerRealizeNode default ok (though does not visit producer)

  Stmt VisitStmt_(const PrefetchNode* prefetch_node) final {
    Prefetch new_prefetch = Downcast<Prefetch>(StmtExprMutator::VisitStmt_(prefetch_node));
    Buffer new_buffer = Subst(new_prefetch->buffer.get());
    if (!new_buffer.same_as(new_prefetch->buffer)) {
      return Prefetch(new_buffer, prefetch_node->bounds, prefetch_node->span);
    }
    return std::move(new_prefetch);
  }

  // SeqStmtNode default ok
  // EvaluateNode default ok

  BufferRegion VisitItem(const BufferRegionNode* buffer_region_node) {
    Buffer new_buffer = Subst(buffer_region_node->buffer.get());
    if (!new_buffer.same_as(buffer_region_node->buffer)) {
      return BufferRegion(new_buffer, buffer_region_node->region);
    }
    return GetRef<BufferRegion>(buffer_region_node);
  }

  MatchBufferRegion VisitItem(const MatchBufferRegionNode* match_buffer_region_node) {
    // The source field has a referential occurrence of the  buffer. Apply the buffer substitution
    // to that.
    BufferRegion new_source = VisitItem(match_buffer_region_node->source.get());
    // The buffer field however is a definitional occurrence, aliased on top of the source.
    // Transfer any memory scope from the source to the destination.
    Optional<VirtualDevice> opt_virtual_device = GetBufferConstraint(new_source->buffer);
    tir::Buffer new_buffer;
    if (opt_virtual_device.defined()) {
      new_buffer = RewriteBuffer(match_buffer_region_node->buffer, opt_virtual_device.value());
    } else {
      new_buffer = match_buffer_region_node->buffer;
    }
    if (!new_buffer.same_as(match_buffer_region_node->buffer) ||
        !new_source.same_as(match_buffer_region_node->source)) {
      return MatchBufferRegion(new_buffer, new_source);
    }
    return GetRef<MatchBufferRegion>(match_buffer_region_node);
  }

  template <typename T>
  Array<T> VisitItems(const Array<T>& items) {
    return items.Map([this](T item) -> T { return VisitItem(item.get()); });
  }

  Stmt VisitStmt_(const BlockNode* block_node) final {
    Block new_block = Downcast<Block>(StmtExprMutator::VisitStmt_(block_node));
    Array<BufferRegion> new_reads = VisitItems(new_block->reads);
    Array<BufferRegion> new_writes = VisitItems(new_block->writes);
    // TODO(mbs): What memory scope should we assign to the new buffers?
    Array<MatchBufferRegion> new_match_buffers = VisitItems(new_block->match_buffers);
    if (!new_reads.same_as(new_block->reads) || new_writes.same_as(new_block->writes) ||
        new_match_buffers.same_as(new_block->match_buffers)) {
      return Block(new_block->iter_vars, std::move(new_reads), std::move(new_writes),
                   new_block->name_hint, new_block->body, new_block->init, new_block->alloc_buffers,
                   std::move(new_match_buffers), new_block->annotations, new_block->span);
    }
    return std::move(new_block);
  }

  // BlockRealizeNode default ok

  /*! Applies \p var_subst_ substitution to \p var_node. */
  Var Subst(const VarNode* var_node) const {
    auto itr = var_subst_.find(var_node);
    return itr == var_subst_.end() ? GetRef<Var>(var_node) : itr->second;
  }

  /*! Applies \p buffer_subst_ substitution to \p buffer. */
  Buffer Subst(const BufferNode* buffer_node) const {
    auto itr = buffer_subst_.find(buffer_node);
    return itr == buffer_subst_.end() ? GetRef<Buffer>(buffer_node) : itr->second;
  }

  /*!
   * \brief Rewrites \p buffer so as to follow the constraints in \p virtual_device
   * (currently just memory scope).
   *
   * Updates both the var_subst_ and buffer_subst_ to capture the rewrite, but
   * also returns the new buffer.
   */
  Buffer RewriteBuffer(const Buffer& buffer, const VirtualDevice& virtual_device) {
    ICHECK(buffer->data->type_annotation.defined());
    const auto* pointer_type_node = buffer->data->type_annotation.as<PointerTypeNode>();
    ICHECK(pointer_type_node);
    if (pointer_type_node->storage_scope == virtual_device->memory_scope) {
      // No change.
      return buffer;
    }
    PointerType new_pointer_type(pointer_type_node->element_type, virtual_device->memory_scope);
    Var new_data(buffer->data->name_hint, new_pointer_type, buffer->data->span);
    var_subst_.emplace(buffer->data.get(), new_data);
    Buffer new_buffer = buffer;
    new_buffer.CopyOnWrite()->data = new_data;
    buffer_subst_.emplace(buffer.get(), new_buffer);
    return new_buffer;
  }

  /*!
   * \brief Returns the VirtualDevice capturing any memory scope in \p buffer. Returns nullptr if
   * buffer's data var does not have a type annotation of \p PointerType. Returns the fully
   * unconstrained \p VirtualDevice if no memory scope is given.
   */
  static Optional<VirtualDevice> GetBufferConstraint(const tir::Buffer& buffer) {
    const auto* pointer_type_node = PointerInBuffer(buffer);
    return pointer_type_node == nullptr
               ? Optional<VirtualDevice>()
               : VirtualDevice::ForMemoryScope(pointer_type_node->storage_scope);
  }

  /*!
   * \brief Maps each \p Buffer::data \p Var to its constrained equivalent.
   */
  std::unordered_map<const VarNode*, Var> var_subst_;

  /*!
   * \brief Maps each \p Buffer to its constrained equivalent.
   */
  std::unordered_map<const BufferNode*, Buffer> buffer_subst_;
};

}  // namespace

Array<VirtualDevice> GetPrimFuncArgAndResultConstraints(const tir::PrimFunc& prim_func,
                                                        const FuncType& relay_func_type) {
  // Build the implied domain (in terms of the function's Relay type) implied by any memory scope
  // constrains in the function's buffers, for both arguments and results.
  Array<VirtualDevice> virtual_devices;
  virtual_devices.reserve(relay_func_type->arg_types.size() + 1);

  // For each Relay function parameter...
  size_t current_primfunc_param_index = 0;
  for (const auto& param_type : relay_func_type->arg_types) {
    VirtualDevice param_virtual_device =
        ConsistentParamConstraint(prim_func, param_type, &current_primfunc_param_index);
    virtual_devices.push_back(param_virtual_device);
  }

  // For the Relay function result...
  const Type& ret_type = relay_func_type->ret_type;
  VirtualDevice ret_virtual_device =
      ConsistentParamConstraint(prim_func, ret_type, &current_primfunc_param_index);
  virtual_devices.push_back(ret_virtual_device);

  // Make sure all parameters of the prim_func have been accounted for.
  CheckNoRemainingPointerParams(prim_func, &current_primfunc_param_index);

  return virtual_devices;
}

TVM_REGISTER_GLOBAL("tir.analysis.GetPrimFuncArgAndResultMemoryConstraints")
    .set_body_typed([](const PrimFunc& prim_func, const FuncType& relay_func_type) {
      Array<String> memory_scopes;
      memory_scopes.reserve(relay_func_type->type_params.size() + 1);
      for (const auto& virtual_device :
           GetPrimFuncArgAndResultConstraints(prim_func, relay_func_type)) {
        memory_scopes.push_back(virtual_device->memory_scope);
      }
      return memory_scopes;
    });

PrimFunc ApplyPrimFuncArgAndResultConstraints(
    const PrimFunc& prim_func, const FuncType& relay_func_type,
    const Array<VirtualDevice>& arg_and_result_virtual_devices) {
  return ApplyDeviceConstraintsMutator().Rewrite(prim_func, relay_func_type,
                                                 arg_and_result_virtual_devices);
}

TVM_REGISTER_GLOBAL("tir.analysis.ApplyPrimFuncArgAndResultMemoryConstraints")
    .set_body_typed([](const PrimFunc& prim_func, const FuncType& relay_func_type,
                       const Array<String>& arg_and_result_memory_scopes) {
      Array<VirtualDevice> virtual_devices;
      virtual_devices.reserve(arg_and_result_memory_scopes.size());
      for (const auto& memory_scope : arg_and_result_memory_scopes) {
        virtual_devices.push_back(VirtualDevice::ForMemoryScope(memory_scope));
      }
      return ApplyPrimFuncArgAndResultConstraints(prim_func, relay_func_type, virtual_devices);
    });

}  // namespace tir
}  // namespace tvm
