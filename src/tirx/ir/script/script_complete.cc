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
 * \file tirx/ir/script/script_complete.cc
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */

#include "./script_complete.h"

#include <tvm/arith/int_set.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>

#include <utility>

namespace tvm {
namespace tirx {

/*! \brief Generate surrounding loops automatically */
class ScriptCompleter : public s_tir::StmtExprMutator {
 public:
  using s_tir::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprNode>()) return ffi::Unchanged();
    return s_tir::StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit ScriptCompleter(ffi::Map<Var, BufferVar>* buffer_var_map, bool s_tir = false)
      : buffer_var_map_(buffer_var_map), s_tir_(s_tir) {}

 private:
  ffi::Map<Var, BufferVar>* buffer_var_map_;
  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockRealizeNode* op, InplaceMode inplace_mode) final {
    for (const PrimExpr& value : op->iter_values) {
      PrimType value_ty = value.ty();
      TVM_FFI_ICHECK(value_ty.code() == DLDataTypeCode::kDLInt)
          << "BlockRealize iter_value expected a IntImm, but got " << value_ty->dtype;
    }
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const s_tir::SBlockNode* op, InplaceMode inplace_mode) final {
    // Buffers allocated in the block can be accessed by its body.
    for (const auto& alloc_buffer : op->alloc_buffers) {
      buffer_var_map_->Set(alloc_buffer.var(), alloc_buffer);
    }
    for (const auto& match_buffer : op->match_buffers) {
      const BufferVar& target_buffer = match_buffer->buffer;
      buffer_var_map_->Set(target_buffer.var(), target_buffer);
    }

    bool is_root_block = this->is_root_block_;
    this->is_root_block_ = false;
    s_tir::SBlock block = s_tir::StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<s_tir::SBlock>();
    this->is_root_block_ = is_root_block;

    // Remove buffers allocated inside block to detect its access region
    for (const auto& alloc_buffer : op->alloc_buffers) {
      buffer_var_map_->erase(alloc_buffer.var());
    }
    for (const auto& match_buffer : op->match_buffers) {
      const BufferVar& target_buffer = match_buffer->buffer;
      buffer_var_map_->erase(target_buffer.var());
    }
    // Get access detection mask
    // 0 for provided region, 1 and 3 for need detect read, 2 and 3 for need detect write
    int mask = 0;
    auto it = op->annotations.find(s_tir::attr::script_parsing_detect_access);
    if (it != op->annotations.end()) {
      mask = (*it).second.as_or_throw<IntImm>()->value.as<int>().value();
    }
    // ignore root block or blocks which already has reads/writes regions
    if (mask != 0 && s_tir_) {
      auto access_region = GetSBlockAccessRegion(block, *buffer_var_map_);
      const ffi::Array<TensorRegion>& reads = access_region[0];
      const ffi::Array<TensorRegion>& writes = access_region[1];
      const ffi::Array<TensorRegion>& opaque = access_region[2];
      TVM_FFI_CHECK(opaque.empty(), ValueError)
          << "Can not auto detect buffer access region from tirx.Load, tirx.Store or "
             "direct access by buffer data. Please annotation the access region manually";
      auto* n = block.CopyOnWrite();
      if (!is_root_block) {
        if (mask & 1) n->reads = reads;
        if (mask & 2) n->writes = writes;
      }
      n->annotations = op->annotations;
      n->annotations.erase(s_tir::attr::script_parsing_detect_access);
      return block;
    } else {
      return block;
    }
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    // AllocBuffer is flat: register buffer for subsequent siblings
    if (!buffer_var_map_->count(op->buffer.var())) {
      buffer_var_map_->Set(op->buffer.var(), op->buffer);
    }
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) final {
    // DeclBuffer is flat: register buffer for subsequent siblings
    if (!buffer_var_map_->count(op->buffer.var())) {
      buffer_var_map_->Set(op->buffer.var(), op->buffer);
    }
    return s_tir::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  bool is_root_block_ = true;
  bool s_tir_ = false;
};

PrimFunc ScriptComplete(PrimFunc func, const ffi::Array<BufferVar>& root_allocates, bool s_tir) {
  ffi::Map<Var, BufferVar> buffer_var_map;
  for (const Var& param : func->params) {
    if (auto buffer = param.as<BufferVar>()) {
      buffer_var_map.Set(buffer.value().var(), buffer.value());
    }
  }
  for (const auto& alloc : root_allocates) {
    buffer_var_map.Set(alloc.var(), alloc);
  }

  Stmt res = func->body;

  // Generate root block automatically.  This is done before
  // ScriptCompleter, in order to fill the root block's T.reads() and
  // T.writes() annotations, as if it had been explicitly written.
  bool should_insert_root = [&]() -> bool {
    if (root_allocates.size()) {
      return true;
    }
    auto* block_realize = func->body.as<s_tir::SBlockRealizeNode>();
    if (block_realize && block_realize->block->iter_vars.size()) {
      return true;
    }
    if (!block_realize && ContainsNode<s_tir::SBlockRealizeNode>(func->body)) {
      return true;
    }
    return false;
  }();

  if (s_tir && should_insert_root) {
    s_tir::SBlock root_block({}, {}, {}, "root", std::move(res), std::nullopt, root_allocates);
    res = s_tir::SBlockRealize({}, IntImm::Bool(true), std::move(root_block));
  }

  // generate surrounding loops automatically
  auto script_completer = ffi::make_object<ScriptCompleter>(&buffer_var_map, s_tir);
  res = script_completer->Mutate(res, InplaceMode::kAllow).ValueOrUnchanged(std::move(res));

  if (func->body.same_as(res)) {
    return func;
  } else {
    auto fptr = func.CopyOnWrite();
    fptr->body = res;
    return func;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.Complete", ScriptComplete);
}

}  // namespace tirx
}  // namespace tvm
