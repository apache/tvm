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
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

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

  static Function Transform(const Function& func) {
    auto generator = ffi::make_object<RenewDefMutator>();
    // Establish explicit parameters and symbols in their types before visiting uses.
    for (const Var& param : func->params) {
      if (param->ty.as<PrimTypeNode>()) generator->ReDefineVar(param);
    }
    ffi::Array<Var> params = func->params.Map([&](const Var& param) {
      auto mapped = generator->VarRemapGet(param);
      return mapped != nullptr ? mapped.as_or_throw<Var>() : generator->ReDefineVar(param);
    });
    // Remap result type and body to the renewed parameter identities.
    Type ret_type = generator->Mutate(func->ret_type, InplaceMode::kDisallow)
                        .as_or_throw<UnchangedOr<Type>>()
                        .ValueOrUnchanged(func->ret_type);
    auto body = generator->Mutate(func->body).ValueOrUnchanged(func->body);
    // Recreate function
    return Function(params, body, ret_type, func->attrs, func->span);
  }

 private:
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    // Result-type patterns may introduce metadata symbols. Renew those before
    // rewriting the RHS; the Bind's own definition is published only afterwards.
    WithDefRegionKind(kTVMFFIDefRegionKindPattern,
                      [&] { return Mutate(op->value->ty, InplaceMode::kDisallow); });
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  STMT_REGENERATE_VAR_DEF(ForNode, loop_var);

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final {
    Var var = ffi::GetRef<Var>(op);
    if (auto mapped = VarRemapGet(var); mapped != nullptr) {
      return mapped.as_or_throw<Expr>();
    }
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return ReDefineVar(var);
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) final {
    // Step 0. Re-define Itervars
    ffi::Array<IterVar> iter_vars =
        op->iter_vars.Map(std::bind(&RenewDefMutator::VisitIterVar, this, std::placeholders::_1));

    // Step 1. Re-define buffers allocated under the block
    ffi::Array<TensorVar> alloc_buffers = op->alloc_buffers.Map([this](const TensorVar& buf) {
      return this->ReDefineVar(buf.var()).as_or_throw<TensorVar>();
    });

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
    if (auto mapped = VarRemapGet(var); mapped != nullptr) return mapped.as_or_throw<Var>();
    Type type = WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&] {
      return Mutate(var->ty, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<Type>>()
          .ValueOrUnchanged(var->ty);
    });
    Var new_var(var->name, type, var->span);
    VarRemapSet(var, new_var);
    return new_var;
  }

  IterVar VisitIterVar(const IterVar& iter_var) {
    auto mapped = VarRemapGet(iter_var);
    if (mapped != nullptr) return mapped.as_or_throw<IterVar>();
    PrimExpr min =
        Mutate(iter_var->dom->min, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->min);
    PrimExpr extent = Mutate(iter_var->dom->extent, InplaceMode::kDisallow)
                          .ValueOrUnchanged(iter_var->dom->extent);
    IterVar new_iter_var(Range::FromMinExtent(min, extent),
                         ReDefineVar(iter_var->var).as_or_throw<PrimVar>(), iter_var->iter_type,
                         iter_var->thread_tag);
    VarRemapSet(iter_var, new_iter_var);
    return new_iter_var;
  }

  MatchBufferRegion VisitMatchBuffer(const MatchBufferRegion& match_buffer) {
    TensorVar buffer = ReDefineVar(match_buffer->buffer.var()).as_or_throw<TensorVar>();
    TensorRegion region = VisitBufferRegion(match_buffer->source);
    return MatchBufferRegion(std::move(buffer), std::move(region));
  }

  TensorRegion VisitBufferRegion(const TensorRegion& buffer_region) {
    return Mutate(buffer_region, InplaceMode::kDisallow)
        .ValueOrUnchanged(buffer_region)
        .as_or_throw<TensorRegion>();
  }
};

Function RenewDefs(const Function& func) { return RenewDefMutator::Transform(func); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.RenewDefs", RenewDefs);
}

}  // namespace s_tir
}  // namespace tvm
