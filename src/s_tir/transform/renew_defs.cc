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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../tirx/ir/functor_common.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

#define STMT_REGENERATE_VAR_DEF(NODE, FIELD)                                                \
  UnchangedOr<Stmt> Mutate_(const NODE* op, InplaceMode inplace_mode) final {               \
    Var new_var = this->ReDefineVar(op->FIELD);                                             \
    Stmt stmt =                                                                             \
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op)); \
    op = stmt.as<NODE>();                                                                   \
    TVM_FFI_ICHECK(op != nullptr);                                                          \
    auto n = ffi::make_object<NODE>(*op);                                                   \
    n->FIELD = std::move(new_var).as_or_throw<std::decay_t<decltype(n->FIELD)>>();          \
    return Stmt(n);                                                                         \
  }

class RenewDefMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static PrimFunc Transform(const PrimFunc& func) {
    auto generator = ffi::make_object<RenewDefMutator>();
    // Redefine scalar parameters first, because they may occur in a buffer
    // parameter's type annotation.
    for (const auto& param : func->params) {
      if (!param.as<BufferVar>()) {
        generator->ReDefineVar(param);
      }
    }
    for (const auto& param : func->params) {
      if (auto opt_buffer = param.as<BufferVar>()) {
        const BufferVar& buffer = opt_buffer.value();
        auto walk_fn = [&generator](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          if (generator->VarRemapGet(var) == nullptr) {
            generator->ReDefineVar(var);
          }
          return ffi::WalkResult::Advance();
        };
        for (const PrimExpr& e : buffer->shape) {
          ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(e, walk_fn);
        }
      }
    }
    // Redefine buffer parameters in order, preserving the original signature.
    // TODO(Siyuan Feng): checking var is used after define
    ffi::Array<Var> params;
    for (const auto& param : func->params) {
      if (auto opt_buffer = param.as<BufferVar>()) {
        params.push_back(generator->DefineBuffer(opt_buffer.value()));
      } else {
        params.push_back(generator->Mutate(param).ValueOrUnchanged(param).as_or_throw<Var>());
      }
    }
    // Visit body
    Stmt body = generator->Mutate(func->body).ValueOrUnchanged(func->body);
    // Recreate function
    return PrimFunc(params, body, func->ret_type, func->attrs, func->span);
  }

 private:
  STMT_REGENERATE_VAR_DEF(BindNode, var);
  STMT_REGENERATE_VAR_DEF(ForNode, loop_var);

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone && VarRemapGet(ffi::AnyView(op)) == nullptr) {
      if (op->ty.as<BufferTypeNode>()) {
        WithDefRegionKind(kTVMFFIDefRegionKindNone, [&] { return DefineBuffer(GetBufferVar(op)); });
      } else {
        ReDefineVar(ffi::GetRef<Var>(op));
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) final {
    // Step 0. Re-define Itervars
    ffi::Array<IterVar> iter_vars =
        op->iter_vars.Map(std::bind(&RenewDefMutator::VisitIterVar, this, std::placeholders::_1));

    // Step 1. Re-define buffers allocated under the block
    ffi::Array<BufferVar> alloc_buffers =
        op->alloc_buffers.Map([this](const BufferVar& buf) { return this->DefineBuffer(buf); });

    // Step 2. Re-define match_buffers
    ffi::Array<MatchBufferRegion> match_buffers = op->match_buffers.Map(
        std::bind(&RenewDefMutator::VisitMatchBuffer, this, std::placeholders::_1));

    // Step 3. Visit body
    ffi::Optional<Stmt> init = std::nullopt;
    if (op->init.has_value()) {
      init = this->Mutate(op->init.value(), inplace_mode).ValueOrUnchanged(op->init.value());
    }
    Stmt body = this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);

    // Step 4. Revisit access region
    ffi::Array<TensorRegion> reads =
        op->reads.Map(std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));
    ffi::Array<TensorRegion> writes =
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

  Var ReDefineVar(const Var& var) {
    Var new_var(var->name, var->ty, var->span);
    this->AddDefRemap(var, new_var);
    return new_var;
  }

  template <typename T>
  void AddDefRemap(const T& source, const T& target) {
    TVM_FFI_ICHECK(VarRemapGet(source) == nullptr);
    VarRemapSet(source, target);
  }

  BufferVar DefineBuffer(const BufferVar& buffer) {
    auto mapped = VarRemapGet(buffer);
    if (mapped != nullptr) return mapped.as_or_throw<BufferVar>();

    auto redefine_if_is_var = [this](const Expr& expr) -> Expr {
      auto mapped = VarRemapGet(expr);
      if (mapped != nullptr) {
        return mapped.as_or_throw<Expr>();
      } else if (auto var = expr.as<Var>()) {
        return this->ReDefineVar(var.value());
      } else {
        return StmtExprMutator::Mutate(ffi::AnyView(expr), InplaceMode::kDisallow)
            .ValueOrUnchanged(expr)
            .as_or_throw<Expr>();
      }
    };

    // shape is USED (references existing definitions like buffer-parameter shape vars),
    // Remap the expression without creating spurious variable definitions.
    auto visit_expr = [this](const PrimExpr& e) -> PrimExpr {
      return this->Mutate(e, InplaceMode::kDisallow).ValueOrUnchanged(e);
    };
    ffi::Array<PrimExpr> shape = buffer->shape.Map(visit_expr);
    // strides/elem_offset may define NEW vars (e.g. in match_buffer),
    // so use redefine_if_is_var to create fresh copies for unknown vars
    ffi::Array<PrimExpr> strides = buffer->strides.Map(
        [&](const PrimExpr& expr) { return redefine_if_is_var(expr).as_or_throw<PrimExpr>(); });
    PrimExpr elem_offset = redefine_if_is_var(buffer->elem_offset).as_or_throw<PrimExpr>();

    auto n = CopyBufferType(buffer);
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    n->elem_offset = std::move(elem_offset);
    BufferVar new_buffer = RebuildBufferVar(buffer, std::move(n));
    this->AddDefRemap(buffer, new_buffer);
    return new_buffer;
  }

  BufferVar UseOrRemapBuffer(const BufferVar& buffer) {
    // If the buffer has been remapped, return the remapped buffer, otherwise,
    // remap it without creating new var definitions.
    auto mapped = VarRemapGet(buffer);
    if (mapped != nullptr) return mapped.as_or_throw<BufferVar>();
    auto visit_expr = [this](const PrimExpr& e) -> PrimExpr {
      return this->Mutate(e, InplaceMode::kDisallow).ValueOrUnchanged(e);
    };
    ffi::Array<PrimExpr> shape = buffer->shape.Map(visit_expr);
    ffi::Array<PrimExpr> strides = buffer->strides.Map(visit_expr);
    PrimExpr elem_offset =
        Mutate(buffer->elem_offset, InplaceMode::kDisallow).ValueOrUnchanged(buffer->elem_offset);

    auto n = CopyBufferType(buffer);
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    n->elem_offset = std::move(elem_offset);
    BufferVar new_buffer = RebuildBufferVar(buffer, std::move(n));
    this->AddDefRemap(buffer, new_buffer);
    return new_buffer;
  }

  IterVar VisitIterVar(const IterVar& iter_var) {
    auto mapped = VarRemapGet(iter_var);
    if (mapped != nullptr) return mapped.as_or_throw<IterVar>();
    PrimExpr min =
        Mutate(iter_var->dom->min, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->min);
    PrimExpr extent = Mutate(iter_var->dom->extent, InplaceMode::kDisallow)
                          .ValueOrUnchanged(iter_var->dom->extent);
    IterVar new_iter_var(Range(min, extent), ReDefineVar(iter_var->var).as_or_throw<PrimVar>(),
                         iter_var->iter_type, iter_var->thread_tag);
    this->AddDefRemap(iter_var, new_iter_var);
    return new_iter_var;
  }

  MatchBufferRegion VisitMatchBuffer(const MatchBufferRegion& match_buffer) {
    BufferVar buffer = DefineBuffer(match_buffer->buffer);
    TensorRegion region = VisitBufferRegion(match_buffer->source);
    return MatchBufferRegion(std::move(buffer), std::move(region));
  }

  Range VisitRange(const Range& range) {
    auto min_result = Mutate(range->min, InplaceMode::kDisallow);
    bool min_unchanged = min_result.UnchangedOrSameAs(range->min);
    PrimExpr min = std::move(min_result).ValueOrUnchanged(range->min);
    auto extent_result = Mutate(range->extent, InplaceMode::kDisallow);
    bool extent_unchanged = extent_result.UnchangedOrSameAs(range->extent);
    PrimExpr extent = std::move(extent_result).ValueOrUnchanged(range->extent);
    if (min_unchanged && extent_unchanged) {
      return range;
    } else {
      return Range::FromMinExtent(std::move(min), std::move(extent));
    }
  }

  TensorRegion VisitBufferRegion(const TensorRegion& buffer_region) {
    BufferVar buffer = UseOrRemapBuffer(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>());
    ffi::Array<Range> region = buffer_region->region.Map(
        std::bind(&RenewDefMutator::VisitRange, this, std::placeholders::_1));
    if (buffer.same_as(buffer_region->source.as_or_throw<tvm::tirx::BufferVar>()) &&
        region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(std::move(buffer), std::move(region));
    }
  }
};

PrimFunc RenewDefs(const PrimFunc& func) { return RenewDefMutator::Transform(func); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.RenewDefs", RenewDefs);
}

}  // namespace s_tir
}  // namespace tvm
