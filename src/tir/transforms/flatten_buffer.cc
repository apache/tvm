/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file flatten_buffer.cc
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"

namespace tvm {
namespace tir {

PrimExpr BufferArea(const Buffer& buffer) {
  PrimExpr area = Integer(1);
  for (const PrimExpr& dim : buffer->shape) {
    area = area * dim;
  }
  return area;
}

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  static Stmt Flatten(const PrimFunc& f) { return BufferFlattener().VisitStmt(f->body); }

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty()) << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
                                       "call pass ConvertBlocksToOpaque before.";
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(op->block));
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    // Step 3. Handle allocations in reverse order
    for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
      const Buffer& buffer = new_block->alloc_buffers[i - 1];
      body = MakeAllocStmt(buffer, std::move(body));
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }
    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);
    // Step 3. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body));
    }
    // Step 4. Handle annotations
    for (const auto& annotation : op->annotations) {
      const String& ann_key = annotation.first;
      const ObjectRef& ann_value = annotation.second;
      if (attr::IsPragmaKey(ann_key)) {
        body = AttrStmt(op->loop_var, ann_key, Downcast<PrimExpr>(ann_value), std::move(body));
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return store->buffer.vstore(store->indices, store->value);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return std::move(var);
    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = Cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return load->buffer.vload(load->indices, load->dtype);
  }

  static Stmt MakeAllocStmt(const Buffer& buffer, Stmt body) {
    String storage_scope = buffer.scope();
    PrimExpr area = BufferArea(buffer);
    body = Allocate(buffer->data, buffer->dtype, {area}, const_true(), std::move(body));
    return body;
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, String thread_tag,
                               Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/std::move(var),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> unit_loop_vars_;
};

PrimFunc FlattenBuffer(PrimFunc f) {
  PrimFuncNode* fptr = f.CopyOnWrite();
  fptr->body = BufferFlattener::Flatten(f);
  return f;
}

namespace transform {

Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return FlattenBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FlattenBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FlattenBuffer").set_body_typed(FlattenBuffer);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
