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
    // Redefine buffers
    Map<tir::Var, Buffer> buffer_map;
    for (const auto& kv : func->buffer_map) {
      const Var& param = kv.first;
      const Buffer& buffer = kv.second;
      Var new_param = Downcast<Var>(generator.VisitExpr(param));
      Buffer new_buffer = generator.VisitBuffer(buffer, true);
      buffer_map.Set(new_param, new_buffer);
    }
    // Visit body
    Stmt body = generator(func->body);
    // Recreate function
    auto n = make_object<PrimFuncNode>(*func.get());
    n->params = std::move(params);
    n->buffer_map = std::move(buffer_map);
    n->body = std::move(body);
    return PrimFunc(n);
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
    Array<IterVar> iter_vars = MutateArray(
        op->iter_vars, std::bind(&RenewDefMutator::VisitIterVar, this, std::placeholders::_1));

    // Step 1. Re-define buffers allocate under the block
    Array<Buffer> alloc_buffers = MutateArray(
        op->alloc_buffers,
        std::bind(&RenewDefMutator::VisitBuffer, this, std::placeholders::_1, /*define=*/true));

    // Step 2. Re-define match_buffers
    Array<MatchBufferRegion> match_buffers =
        MutateArray(op->match_buffers,
                    std::bind(&RenewDefMutator::VisitMatchBuffer, this, std::placeholders::_1));

    // Step 3. Visit body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op);

    // Step 4. Revisit access region
    Array<BufferRegion> reads = MutateArray(
        op->reads, std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));
    Array<BufferRegion> writes = MutateArray(
        op->writes, std::bind(&RenewDefMutator::VisitBufferRegion, this, std::placeholders::_1));

    // Step 5. Regenerate block. Since the defs are changed, we need to create a new block
    auto n = make_object<BlockNode>(*op);
    n->iter_vars = std::move(iter_vars);
    n->alloc_buffers = std::move(alloc_buffers);
    n->match_buffers = std::move(match_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);

    return Stmt(n);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op != nullptr);
    Buffer buffer = VisitDeclBuffer(op->buffer);
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
    Buffer buffer = VisitDeclBuffer(op->buffer);
    if (buffer.same_as(op->buffer)) {
      return expr;
    } else {
      auto n = make_object<BufferLoadNode>(*op);
      n->buffer = std::move(buffer);
      return BufferLoad(n);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
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
      } else if (const VarNode* var = expr.as<VarNode>()) {
        return this->ReDefineVar(GetRef<Var>(var));
      } else {
        return ExprMutator::VisitExpr(expr);
      }
    };

    // update data
    Var data = Downcast<Var>(redefine_if_is_var(buffer->data));
    // update shape
    Array<PrimExpr> shape = MutateArray(buffer->shape, redefine_if_is_var);
    // update strides
    Array<PrimExpr> strides = MutateArray(buffer->strides, redefine_if_is_var);
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

  Buffer VisitDeclBuffer(const Buffer& buffer) {
    // Due to a recent PR, we can allow undefined buffer appearing in BufferLoad/Store.
    // We need to remap them but will not create new var
    auto it = remap_.find(buffer);
    if (it != remap_.end()) {
      return Downcast<Buffer>((*it).second);
    }
    Var data = Downcast<Var>(VisitExpr(buffer->data));
    Array<PrimExpr> shape = MutateArray(
        buffer->shape, std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
    Array<PrimExpr> strides = MutateArray(
        buffer->strides, std::bind(&RenewDefMutator::VisitExpr, this, std::placeholders::_1));
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
    Array<Range> region =
        MutateArray(buffer_region->region,
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
