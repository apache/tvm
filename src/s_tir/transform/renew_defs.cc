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
 * \file renew_defs.cc
 * \brief Renew the definition nodes for a TIR, including Var, Buffer and IterVar.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../tirx/ir/functor_common.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

#define STMT_REGENERATE_VAR_DEF(NODE, FIELD)     \
  Stmt VisitStmt_(const NODE* op) final {        \
    Var new_var = this->ReDefineVar(op->FIELD);  \
    Stmt stmt = StmtExprMutator::VisitStmt_(op); \
    op = stmt.as<NODE>();                        \
    TVM_FFI_ICHECK(op != nullptr);               \
    auto n = ffi::make_object<NODE>(*op);        \
    n->FIELD = std::move(new_var);               \
    return Stmt(n);                              \
  }

class RenewDefMutator : public StmtExprMutator {
 public:
  static PrimFunc Transform(const PrimFunc& func) {
    RenewDefMutator generator;
    // Redefine params
    ffi::Array<Var> params;
    for (const auto& param : func->params) {
      params.push_back(generator.ReDefineVar(param));
    }
    for (const auto& param : func->params) {
      if (param->dtype.is_handle()) {
        const Buffer& buffer = func->buffer_map.at(param);
        for (const PrimExpr& e : buffer->shape) {
          if (const auto* v = e.as<VarNode>()) {
            if (generator.remap_.count(ffi::GetRef<Var>(v)) == 0) {
              generator.ReDefineVar(ffi::GetRef<Var>(v));
            }
          }
        }
      }
    }
    // Redefine buffers in order
    // TODO(Siyuan Feng): checking var is used after define
    ffi::Map<tirx::Var, Buffer> buffer_map;
    for (const auto& param : func->params) {
      if (param->dtype.is_handle()) {
        const Buffer& buffer = func->buffer_map.at(param);
        Var new_param = Downcast<Var>(generator.VisitExpr(param));
        Buffer new_buffer = generator.DefineBuffer(buffer);
        buffer_map.Set(new_param, new_buffer);
      }
    }
    // Visit body
    Stmt body = generator(func->body);
    // Recreate function
    return PrimFunc(params, body, func->ret_type, buffer_map, func->attrs, func->span);
  }

 private:
  Stmt operator()(Stmt stmt) {
    // override StmtMutator::operator() to disable copy_on_write
    // Since this pass tries to explicit create a new function rather than update the existing one
    allow_copy_on_write_ = false;
    return VisitStmt(stmt);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) final {
    auto it = remap_.find(expr);
    if (it != remap_.end()) {
      return Downcast<PrimExpr>((*it).second);
    } else {
      return ExprMutator::VisitExpr(expr);
    }
  }

 private:
  STMT_REGENERATE_VAR_DEF(BindNode, var);
  STMT_REGENERATE_VAR_DEF(ForNode, loop_var);

  // Override VisitBufferDef to create fresh buffer copies at definition sites
  // (AllocBuffer, DeclBuffer, SBlock alloc_buffers, match_buffers)
  Buffer VisitBufferDef(const Buffer& buffer, bool alloc_data) final {
    return DefineBuffer(buffer);
  }

  // Override VisitBufferUse to remap buffers at use sites
  // (BufferStore, BufferLoad, SBlock reads/writes)
  Buffer VisitBufferUse(const Buffer& buffer) final { return UseOrRemapBuffer(buffer); }

  Stmt VisitStmt_(const SBlockNode* op) final {
    // Step 0. Re-define Itervars
    ffi::Array<IterVar> iter_vars =
        op->iter_vars.Map(std::bind(&RenewDefMutator::VisitIterVar, this, std::placeholders::_1));

    // Step 1. Re-define buffers allocated under the block
    ffi::Array<Buffer> alloc_buffers =
        op->alloc_buffers.Map([this](const Buffer& buf) { return this->DefineBuffer(buf); });

    // Step 2. Re-define match_buffers
    ffi::Array<MatchBufferRegion> match_buffers = op->match_buffers.Map(
        std::bind(&RenewDefMutator::VisitMatchBuffer, this, std::placeholders::_1));

    // Step 3. Visit body
    ffi::Optional<Stmt> init = std::nullopt;
    if (op->init.defined()) {
      init = this->VisitStmt(op->init.value());
    }
    Stmt body = this->VisitStmt(op->body);

    // Step 4. Revisit access region
    ffi::Array<BufferRegion> reads =
        op->reads.Map(std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));
    ffi::Array<BufferRegion> writes =
        op->writes.Map(std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));

    // Step 5. Regenerate block. Since the defs are changed, we need to create a new block
    auto n = ffi::make_object<SBlockNode>(*op);
    n->iter_vars = std::move(iter_vars);
    n->alloc_buffers = std::move(alloc_buffers);
    n->match_buffers = std::move(match_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);

    return Stmt(n);
  }

 private:
  Var ReDefineVar(const Var& var) {
    Var new_var = Var(ffi::make_object<VarNode>(*var.get()));
    this->AddDefRemap(var, new_var);
    return new_var;
  }

  template <typename T>
  void AddDefRemap(const T& source, const T& target) {
    TVM_FFI_ICHECK(remap_.count(source) == 0);
    remap_.Set(source, target);
  }

  Buffer DefineBuffer(const Buffer& buffer) {
    auto it = remap_.find(buffer);
    if (it != remap_.end()) {
      return Downcast<Buffer>((*it).second);
    }

    auto redefine_if_is_var = [this](const PrimExpr& expr) -> PrimExpr {
      auto it = remap_.find(expr);
      if (it != remap_.end()) {
        return Downcast<PrimExpr>((*it).second);
      } else if (auto var = expr.as<Var>()) {
        return this->ReDefineVar(var.value());
      } else {
        return ExprMutator::VisitExpr(expr);
      }
    };

    // data is DEFINED by this buffer — needs a fresh copy
    Var data = Downcast<Var>(redefine_if_is_var(buffer->data));
    // shape is USED (references existing definitions like buffer_map shape vars),
    // remap via VisitExpr to avoid creating spurious new var definitions
    auto visit_expr = [this](const PrimExpr& e) -> PrimExpr { return this->VisitExpr(e); };
    ffi::Array<PrimExpr> shape = buffer->shape.Map(visit_expr);
    // strides/elem_offset may define NEW vars (e.g. in match_buffer),
    // so use redefine_if_is_var to create fresh copies for unknown vars
    ffi::Array<PrimExpr> strides = buffer->strides.Map(redefine_if_is_var);
    PrimExpr elem_offset = redefine_if_is_var(buffer->elem_offset);

    auto n = ffi::make_object<BufferNode>(*buffer.get());
    n->data = std::move(data);
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    n->elem_offset = std::move(elem_offset);
    Buffer new_buffer(n);
    this->AddDefRemap(buffer, new_buffer);
    return new_buffer;
  }

  Buffer UseOrRemapBuffer(const Buffer& buffer) {
    // If the buffer has been remapped, return the remapped buffer, otherwise,
    // remap it without creating new var definitions.
    auto it = remap_.find(buffer);
    if (it != remap_.end()) {
      return Downcast<Buffer>((*it).second);
    }
    Var data = Downcast<Var>(VisitExpr(buffer->data));
    ffi::Array<PrimExpr> shape =
        buffer->shape.Map(std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
    ffi::Array<PrimExpr> strides =
        buffer->strides.Map(std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
    PrimExpr elem_offset = VisitExpr(buffer->elem_offset);

    auto n = ffi::make_object<BufferNode>(*buffer.get());
    n->data = std::move(data);
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    n->elem_offset = std::move(elem_offset);
    Buffer new_buffer(n);
    this->AddDefRemap(buffer, new_buffer);
    return new_buffer;
  }

  IterVar VisitIterVar(const IterVar& iter_var) {
    auto it = remap_.find(iter_var);
    if (it != remap_.end()) {
      return Downcast<IterVar>((*it).second);
    }
    PrimExpr min = VisitExpr(iter_var->dom->min);
    PrimExpr extent = VisitExpr(iter_var->dom->extent);
    IterVar new_iter_var(Range(min, extent), ReDefineVar(iter_var->var), iter_var->iter_type,
                         iter_var->thread_tag);
    this->AddDefRemap(iter_var, new_iter_var);
    return new_iter_var;
  }

  MatchBufferRegion VisitMatchBuffer(const MatchBufferRegion& match_buffer) {
    Buffer buffer = DefineBuffer(match_buffer->buffer);
    BufferRegion region = VisitBufferRegion(match_buffer->source);
    return MatchBufferRegion(std::move(buffer), std::move(region));
  }

  Range VisitRange(const Range& range) {
    PrimExpr min = VisitExpr(range->min);
    PrimExpr extent = VisitExpr(range->extent);
    if (min.same_as(range->min) && extent.same_as(range->extent)) {
      return range;
    } else {
      return Range::FromMinExtent(std::move(min), std::move(extent));
    }
  }

  BufferRegion VisitBufferRegion(const BufferRegion& buffer_region) {
    Buffer buffer = UseOrRemapBuffer(buffer_region->buffer);
    ffi::Array<Range> region = buffer_region->region.Map(
        std::bind(&RenewDefMutator::VisitRange, this, std::placeholders::_1));
    if (buffer.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(std::move(buffer), std::move(region));
    }
  }

  ffi::Map<ffi::ObjectRef, ffi::ObjectRef> remap_;
};

PrimFunc RenewDefs(const PrimFunc& func) { return RenewDefMutator::Transform(func); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.RenewDefs", RenewDefs);
}

}  // namespace s_tir
}  // namespace tvm
