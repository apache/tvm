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

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../ir/functor_common.h"

namespace tvm {
namespace tir {

#define STMT_REGENERATE_VAR_DEF(NODE, FIELD)     \
  Stmt VisitStmt_(const NODE* op) final {        \
    Var new_var = this->ReDefineVar(op->FIELD);  \
    Stmt stmt = StmtExprMutator::VisitStmt_(op); \
    op = stmt.as<NODE>();                        \
    ICHECK(op != nullptr);                       \
    auto n = make_object<NODE>(*op);             \
    n->FIELD = std::move(new_var);               \
    return Stmt(n);                              \
  }

class RenewDefMutator : public StmtExprMutator {
 public:
  static PrimFunc Transform(const PrimFunc& func) {
    RenewDefMutator generator;
    // Redefine params
    Array<Var> params;
    for (const auto& param : func->params) {
      params.push_back(generator.ReDefineVar(param));
    }
    for (const auto& param : func->params) {
      if (param->dtype.is_handle()) {
        const Buffer& buffer = func->buffer_map.at(param);
        for (const PrimExpr& e : buffer->shape) {
          if (const auto* v = e.as<VarNode>()) {
            if (generator.remap_.count(GetRef<Var>(v)) == 0) {
              generator.ReDefineVar(GetRef<Var>(v));
            }
          }
        }
      }
    }
    // Redefine buffers in order
    // TODO(Siyuan Feng): checking var is used after define
    Map<tir::Var, Buffer> buffer_map;
    for (const auto& param : func->params) {
      if (param->dtype.is_handle()) {
        const Buffer& buffer = func->buffer_map.at(param);
        Var new_param = Downcast<Var>(generator.VisitExpr(param));
        Buffer new_buffer = generator.VisitBuffer(buffer, true);
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
    // Since this pass tries to explict create a new function rather than update the existing one
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
  STMT_REGENERATE_VAR_DEF(LetStmtNode, var);
  STMT_REGENERATE_VAR_DEF(AllocateNode, buffer_var);
  STMT_REGENERATE_VAR_DEF(AllocateConstNode, buffer_var);
  STMT_REGENERATE_VAR_DEF(ForNode, loop_var);

  Stmt VisitStmt_(const BlockNode* op) final {
    // Step 0. Re-define Itervars
    Array<IterVar> iter_vars =
        op->iter_vars.Map(std::bind(&RenewDefMutator::VisitIterVar, this, std::placeholders::_1));

    // Step 1. Re-define buffers allocate under the block
    Array<Buffer> alloc_buffers = op->alloc_buffers.Map(
        std::bind(&RenewDefMutator::VisitBuffer, this, std::placeholders::_1, /*define=*/true));

    // Step 2. Re-define match_buffers
    Array<MatchBufferRegion> match_buffers = op->match_buffers.Map(
        std::bind(&RenewDefMutator::VisitMatchBuffer, this, std::placeholders::_1));

    // Step 3. Visit body
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = this->VisitStmt(op->init.value());
    }
    Stmt body = this->VisitStmt(op->body);

    // Step 4. Revisit access region
    Array<BufferRegion> reads =
        op->reads.Map(std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes =
        op->writes.Map(std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));

    // Step 5. Regenerate block. Since the defs are changed, we need to create a new block
    auto n = make_object<BlockNode>(*op);
    n->iter_vars = std::move(iter_vars);
    n->alloc_buffers = std::move(alloc_buffers);
    n->match_buffers = std::move(match_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);

    return Stmt(n);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);
    Buffer buffer = VisitDeclOrRemapBuffer(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return stmt;
    } else {
      auto n = make_object<BufferStoreNode>(*op);
      n->buffer = std::move(buffer);
      return BufferStore(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op != nullptr);
    Buffer buffer = VisitDeclOrRemapBuffer(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return expr;
    } else {
      auto n = make_object<BufferLoadNode>(*op);
      n->buffer = std::move(buffer);
      return BufferLoad(n);
    }
  }

 private:
  Var ReDefineVar(const Var& var) {
    Var new_var = Var(make_object<VarNode>(*var.get()));
    this->AddDefRemap(var, new_var);
    return new_var;
  }

  template <typename T>
  void AddDefRemap(const T& source, const T& target) {
    ICHECK(remap_.count(source) == 0);
    remap_.Set(source, target);
  }

  Buffer VisitBuffer(const Buffer& buffer, bool define = false) {
    auto it = remap_.find(buffer);
    if (it != remap_.end()) {
      return Downcast<Buffer>((*it).second);
    }
    ICHECK(define);

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

    // update data
    Var data = Downcast<Var>(redefine_if_is_var(buffer->data));
    // update shape
    Array<PrimExpr> shape = buffer->shape.Map(redefine_if_is_var);
    // update strides
    Array<PrimExpr> strides = buffer->strides.Map(redefine_if_is_var);
    // update elem_offset
    PrimExpr elem_offset = redefine_if_is_var(buffer->elem_offset);

    auto n = make_object<BufferNode>(*buffer.get());
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

  Buffer VisitDeclOrRemapBuffer(const Buffer& buffer) {
    // If the buffer has been remapped, return the remapped buffer, otherwise,
    // return the declared one.
    // Due to a recent PR, we can allow undefined buffer appearing in BufferLoad/Store. We need
    // to remap them but will not create new var
    auto it = remap_.find(buffer);
    if (it != remap_.end()) {
      return Downcast<Buffer>((*it).second);
    }
    Var data = Downcast<Var>(VisitExpr(buffer->data));
    Array<PrimExpr> shape =
        buffer->shape.Map(std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
    Array<PrimExpr> strides =
        buffer->strides.Map(std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
    PrimExpr elem_offset = VisitExpr(buffer->elem_offset);

    auto n = make_object<BufferNode>(*buffer.get());
    n->data = std::move(data);
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    n->elem_offset = std::move(elem_offset);
    Buffer new_buffer(n);
    this->AddDefRemap(buffer, new_buffer);
    return new_buffer;
  }

  MatchBufferRegion VisitMatchBuffer(const MatchBufferRegion& match_buffer) {
    Buffer buffer = VisitBuffer(match_buffer->buffer, /*define=*/true);
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
    Buffer buffer = VisitBuffer(buffer_region->buffer);
    Array<Range> region = buffer_region->region.Map(
        std::bind(&RenewDefMutator::VisitRange, this, std::placeholders::_1));
    if (buffer.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
      return buffer_region;
    } else {
      return BufferRegion(std::move(buffer), std::move(region));
    }
  }

  Map<ObjectRef, ObjectRef> remap_;
};

PrimFunc RenewDefs(const PrimFunc& func) { return RenewDefMutator::Transform(func); }

TVM_REGISTER_GLOBAL("tir.RenewDefs").set_body_typed(RenewDefs);

}  // namespace tir
}  // namespace tvm
