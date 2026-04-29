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
 * \file stmt_functor.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>

#include "data_type_rewriter.h"
#include "functor_common.h"

namespace tvm {
namespace tirx {

void StmtVisitor::VisitStmt_(const BindNode* op) {
  // Bind has no body -- only visit the value expression.
  this->VisitExpr(op->value);
}

void StmtVisitor::VisitStmt_(const AttrStmtNode* op) {
  this->VisitExpr(op->value);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const ForNode* op) {
  this->VisitExpr(op->min);
  this->VisitExpr(op->extent);
  if (op->step.has_value()) {
    this->VisitExpr(*op->step);
  }
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const WhileNode* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitBufferDef(const Buffer& buffer, bool alloc_data) {
  for (const auto& e : buffer->shape) this->VisitExpr(e);
  for (const auto& e : buffer->strides) this->VisitExpr(e);
  this->VisitExpr(buffer->elem_offset);
}

// Default VisitBufferUse is empty: buffer fields (shape, strides, elem_offset)
// are visited at the definition site (VisitBufferDef) and should not be
// re-visited at each use site, as the use site may be in a different scope
// where the buffer's shape variables are not defined.
void StmtVisitor::VisitBufferUse(const Buffer& buffer) {}

void StmtExprVisitor::VisitExpr_(const BufferLoadNode* op) {
  this->VisitBufferUse(op->buffer);
  ExprVisitor::VisitExpr_(op);
}

void StmtVisitor::VisitStmt_(const AllocBufferNode* op) {
  this->VisitBufferDef(op->buffer, /*alloc_data=*/true);
}

void StmtVisitor::VisitStmt_(const DeclBufferNode* op) {
  this->VisitBufferDef(op->buffer, /*alloc_data=*/false);
}

void StmtVisitor::VisitStmt_(const BufferStoreNode* op) {
  this->VisitBufferUse(op->buffer);
  this->VisitExpr(op->value);
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void StmtVisitor::VisitStmt_(const IfThenElseNode* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->then_case);
  if (op->else_case) {
    this->VisitStmt(op->else_case.value());
  }
}

void StmtVisitor::VisitStmt_(const AssertStmtNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->error_kind);
  VisitArray(op->message_parts, [this](const StringImm& e) { this->VisitExpr(e); });
}

void StmtVisitor::VisitStmt_(const SeqStmtNode* op) {
  VisitArray(op->seq, [this](const Stmt& s) { this->VisitStmt(s); });
}

void StmtVisitor::VisitStmt_(const EvaluateNode* op) { this->VisitExpr(op->value); }

void StmtVisitor::VisitStmt_(const SBlockNode* op) {
  auto fvisit_buffer_region = [this](const BufferRegion& s) {
    this->VisitBufferUse(s->buffer);
    for (const auto& range : s->region) {
      this->VisitExpr(range->min);
      this->VisitExpr(range->extent);
    }
  };
  VisitArray(op->iter_vars, [this](const IterVar& iter_var) {
    this->VisitExpr(iter_var->dom->min);
    this->VisitExpr(iter_var->dom->extent);
  });
  VisitArray(op->alloc_buffers,
             [this](const Buffer& buf) { this->VisitBufferDef(buf, /*alloc_data=*/true); });
  VisitArray(op->reads, fvisit_buffer_region);
  VisitArray(op->writes, fvisit_buffer_region);
  VisitArray(op->match_buffers,
             [this, &fvisit_buffer_region](const MatchBufferRegion& match_buffer_region) {
               this->VisitBufferDef(match_buffer_region->buffer, /*alloc_data=*/true);
               fvisit_buffer_region(match_buffer_region->source);
             });
  if (op->init.defined()) {
    this->VisitStmt(op->init.value());
  }
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const SBlockRealizeNode* op) {
  VisitArray(op->iter_values, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->predicate);
  this->VisitStmt(op->block);
}

class StmtMutator::Internal {
 public:
  /*!
   * \brief Mutate array's element by fmutate function.
   *
   * \note Use extra care for copy on write setting.
   *
   * In particular, consider the following case of two reference chains:
   * - strongref0 -> loop0 -> loop1 -> loop2
   * - strongref1 -> loop3 -> loop1 -> loop2
   *
   * Think of the case of calling MutateArray on loop1->loop2(as const reference).
   * When both strongref0 and strongref1 exists, the context does not allow copy
   * on write, even though loop1 uniquely refers to loop2.
   *
   * \param self The pointer to the mutator.
   * \param arr Array to be mutated, const reference is used to allow copy on write
   *            mutation in a recursive visitor.
   * \param fmutate The mutator function.
   * \return The mutated array, a new copy can be created.
   */
  template <typename T, typename F>
  static ffi::Array<T> MutateArray(StmtMutator* self, const ffi::Array<T>& arr, F fmutate) {
    if (self->allow_copy_on_write_ && arr.unique()) {
      // if we allow copy on write, we can directly
      // call the inplace mutate function.
      const_cast<ffi::Array<T>&>(arr).MutateByApply(fmutate);
      return arr;
    } else {
      bool allow_cow = false;
      std::swap(allow_cow, self->allow_copy_on_write_);
      ffi::Array<T> copy = arr.Map(fmutate);
      std::swap(allow_cow, self->allow_copy_on_write_);
      return copy;
    }
  }

  static ffi::Array<IterVar> Mutate(StmtMutator* self, const ffi::Array<IterVar>& arr) {
    auto fmutate = [self](const IterVar& iter_var) {
      PrimExpr min = self->VisitExpr(iter_var->dom->min);
      PrimExpr extent = self->VisitExpr(iter_var->dom->extent);
      if (min.same_as(iter_var->dom->min) && extent.same_as(iter_var->dom->extent)) {
        return iter_var;
      } else {
        return IterVar(Range(min, extent), iter_var->var, iter_var->iter_type,
                       iter_var->thread_tag);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<PrimExpr> Mutate(StmtMutator* self, const ffi::Array<PrimExpr>& arr) {
    auto fmutate = [self](const PrimExpr& e) { return self->VisitExpr(e); };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<Stmt> Mutate(StmtMutator* self, const ffi::Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<Range> Mutate(StmtMutator* self, const ffi::Array<Range>& arr) {
    auto fmutate = [self](const Range& r) {
      PrimExpr min = self->VisitExpr(r->min);
      PrimExpr extent = self->VisitExpr(r->extent);
      if (min.same_as(r->min) && extent.same_as(r->extent)) {
        return r;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<BufferRegion> Mutate(StmtMutator* self, const ffi::Array<BufferRegion>& arr) {
    auto fmutate = [self](const BufferRegion& buffer_region) {
      Buffer new_buf = self->VisitBufferUse(buffer_region->buffer);
      ffi::Array<Range> region = Mutate(self, buffer_region->region);
      if (new_buf.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(std::move(new_buf), std::move(region));
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<MatchBufferRegion> Mutate(StmtMutator* self,
                                              const ffi::Array<MatchBufferRegion>& arr) {
    auto fmutate = [self](const MatchBufferRegion& match_buffer_region) {
      Buffer new_buf = self->VisitBufferDef(match_buffer_region->buffer, /*alloc_data=*/true);
      Buffer new_source_buf = self->VisitBufferUse(match_buffer_region->source->buffer);
      ffi::Array<Range> region = Mutate(self, match_buffer_region->source->region);
      if (new_buf.same_as(match_buffer_region->buffer) &&
          new_source_buf.same_as(match_buffer_region->source->buffer) &&
          region.same_as(match_buffer_region->source->region)) {
        return match_buffer_region;
      } else {
        return MatchBufferRegion(std::move(new_buf),
                                 BufferRegion(std::move(new_source_buf), std::move(region)));
      }
    };
    return MutateArray(self, arr, fmutate);
  }
};

Stmt StmtMutator::VisitStmt_(const BindNode* op) {
  // Bind has no body -- only mutate the value expression.
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AttrStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ForNode* op) {
  PrimExpr min = this->VisitExpr(op->min);
  PrimExpr extent = this->VisitExpr(op->extent);
  ffi::Optional<PrimExpr> step{std::nullopt};
  if (op->step.has_value()) {
    step = this->VisitExpr(*op->step);
  }
  Stmt body = this->VisitStmt(op->body);
  if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body) &&
      step.same_as(op->step)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->step = std::move(step);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const WhileNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt body = this->VisitStmt(op->body);
  if (condition.same_as(op->condition) && body.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Buffer StmtMutator::VisitBufferDef(const Buffer& buffer, bool alloc_data) {
  if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
    return (*it).second;
  }

  // Visit expression fields (shape, strides, elem_offset) but NOT data.
  // data is a Var definition owned by this buffer, not an expression use.
  // Subclasses that need to remap data (e.g., IRSubstitute) can override.
  auto shape = buffer->shape.Map([this](const PrimExpr& e) { return this->VisitExpr(e); });
  auto strides = buffer->strides.Map([this](const PrimExpr& e) { return this->VisitExpr(e); });
  PrimExpr elem_offset = this->VisitExpr(buffer->elem_offset);

  if (shape.same_as(buffer->shape) && strides.same_as(buffer->strides) &&
      elem_offset.same_as(buffer->elem_offset)) {
    return buffer;
  }
  Buffer new_buf = buffer;
  auto* n = new_buf.CopyOnWrite();
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->elem_offset = std::move(elem_offset);
  buffer_remap_.Set(buffer, new_buf);
  return new_buf;
}

Buffer StmtMutator::VisitBufferUse(const Buffer& buffer) {
  if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
    return (*it).second;
  }
  return buffer;
}

PrimExpr StmtExprMutator::VisitExpr_(const BufferLoadNode* op) {
  Buffer new_buf = this->VisitBufferUse(op->buffer);
  PrimExpr expr = ExprMutator::VisitExpr_(op);
  op = expr.as<BufferLoadNode>();
  TVM_FFI_ICHECK(op != nullptr);
  if (!new_buf.same_as(op->buffer)) {
    auto n = ffi::make_object<BufferLoadNode>(*op);
    n->buffer = std::move(new_buf);
    return PrimExpr(n);
  }
  return expr;
}

Stmt StmtMutator::VisitStmt_(const AllocBufferNode* op) {
  Buffer new_buf = this->VisitBufferDef(op->buffer, /*alloc_data=*/true);

  if (new_buf.same_as(op->buffer)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buf);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const DeclBufferNode* op) {
  Buffer new_buf = this->VisitBufferDef(op->buffer, /*alloc_data=*/false);

  if (new_buf.same_as(op->buffer)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buf);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt then_case = this->VisitStmt(op->then_case);
  ffi::Optional<Stmt> else_case = std::nullopt;
  if (op->else_case) {
    else_case = this->VisitStmt(op->else_case.value());
  }
  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BufferStoreNode* op) {
  Buffer new_buf = this->VisitBufferUse(op->buffer);
  PrimExpr value = this->VisitExpr(op->value);
  ffi::Array<PrimExpr> indices = Internal::Mutate(this, op->indices);

  if (new_buf.same_as(op->buffer) && value.same_as(op->value) && indices.same_as(op->indices)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buf);
    n->value = std::move(value);
    n->indices = std::move(indices);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SeqStmtNode* op) {
  ffi::Array<Stmt> seq = Internal::Mutate(this, op->seq);
  if (seq.same_as(op->seq)) {
    return SeqStmt::Flatten(ffi::GetRef<Stmt>(op));
  } else {
    auto node = CopyOnWrite(op);
    node->seq = std::move(seq);
    return SeqStmt::Flatten(SeqStmt(node));
  }
}

// advanced visit function for seqstmt.
Stmt StmtMutator::VisitSeqStmt_(const SeqStmtNode* op, bool flatten_before_visit,
                                std::function<Stmt(const Stmt&)> fmutate) {
  if (flatten_before_visit) {
    // Pass 1, check if we need to flatten.
    bool need_flatten = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      Stmt tmp = (*op)[i];
      if (tmp.as<SeqStmtNode>()) need_flatten = true;
    }
    flatten_before_visit = need_flatten;
  }
  // function to run the visit.
  auto frunvisit = [&](const SeqStmtNode* op) {
    ffi::Array<Stmt> seq = fmutate != nullptr ? Internal::MutateArray(this, op->seq, fmutate)
                                              : Internal::Mutate(this, op->seq);
    if (seq.same_as(op->seq)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->seq = std::move(seq);
      return Stmt(n);
    }
  };
  if (flatten_before_visit) {
    ffi::Array<Stmt> seq;
    SeqStmt::Flattener flattener(&seq);
    flattener(0, op->seq);
    // NOTE: If copy on write is allowed
    // the assignment to seq below will
    // destruct the original seq.
    //
    // Such destruction removes duplicated reference
    // count to children and still enables COW for
    // child Stmt.
    ObjectPtr<SeqStmtNode> n = CopyOnWrite(op);
    n->seq = std::move(seq);
    return frunvisit(n.operator->());
  } else {
    return frunvisit(op);
  }
}

Stmt StmtMutator::VisitStmt_(const AssertStmtNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr error_kind = this->VisitExpr(op->error_kind);
  ffi::Array<StringImm> message_parts = Internal::MutateArray(
      this, op->message_parts,
      [this](const StringImm& e) { return Downcast<StringImm>(this->VisitExpr(e)); });

  if (condition.same_as(op->condition) && error_kind.same_as(op->error_kind) &&
      message_parts.same_as(op->message_parts)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->error_kind = Downcast<StringImm>(std::move(error_kind));
    n->message_parts = std::move(message_parts);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const EvaluateNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SBlockNode* op) {
  ffi::Array<IterVar> iter_vars = Internal::Mutate(this, op->iter_vars);
  ffi::Array<Buffer> alloc_buffers = Internal::MutateArray(
      this, op->alloc_buffers,
      [this](const Buffer& buf) { return this->VisitBufferDef(buf, /*alloc_data=*/true); });
  ffi::Array<BufferRegion> reads = Internal::Mutate(this, op->reads);
  ffi::Array<BufferRegion> writes = Internal::Mutate(this, op->writes);
  ffi::Array<MatchBufferRegion> match_buffers = Internal::Mutate(this, op->match_buffers);
  ffi::Optional<Stmt> init = std::nullopt;
  if (op->init.defined()) {
    init = VisitStmt(op->init.value());
  }
  Stmt body = VisitStmt(op->body);
  if (iter_vars.same_as(op->iter_vars) && alloc_buffers.same_as(op->alloc_buffers) &&
      reads.same_as(op->reads) && writes.same_as(op->writes) && body.same_as(op->body) &&
      init.same_as(op->init) && match_buffers.same_as(op->match_buffers)) {
    return ffi::GetRef<SBlock>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_vars = std::move(iter_vars);
    n->alloc_buffers = std::move(alloc_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);
    n->match_buffers = std::move(match_buffers);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SBlockRealizeNode* op) {
  ffi::Array<PrimExpr> v = Internal::Mutate(this, op->iter_values);
  PrimExpr pred = this->VisitExpr(op->predicate);
  Stmt block = this->VisitStmt(op->block);
  if (v.same_as(op->iter_values) && pred.same_as(op->predicate) && block.same_as(op->block)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_values = std::move(v);
    n->predicate = std::move(pred);
    n->block = Downcast<SBlock>(block);
    return Stmt(n);
  }
}

// Implementations of IRTransform, PostOrderVisit and Substitute
class IRApplyVisit : public StmtExprVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const ObjectRef&)> f) : f_(f) {}

  void VisitExpr(const PrimExpr& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    ExprVisitor::VisitExpr(node);
    f_(node);
  }

  void VisitStmt(const Stmt& node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    StmtVisitor::VisitStmt(node);
    f_(node);
  }

  void VisitBufferDef(const Buffer& buffer, bool alloc_data) override {}
  void VisitBufferUse(const Buffer& buffer) override {}

 private:
  std::function<void(const ObjectRef&)> f_;
  std::unordered_set<const Object*> visited_;
};

void PostOrderVisit(const ObjectRef& node, std::function<void(const ObjectRef&)> fvisit) {
  if (node.as<StmtNode>()) {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<Stmt>(node));
  } else {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<PrimExpr>(node));
  }
}

class IRTransformer final : public StmtExprMutator {
 public:
  IRTransformer(const ffi::Function& f_preorder, const ffi::Function& f_postorder,
                const std::unordered_set<uint32_t>& only_enable)
      : f_preorder_(f_preorder), f_postorder_(f_postorder), only_enable_(only_enable) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    return MutateInternal<Stmt>(stmt, [this](const Stmt& s) { return this->BaseVisitStmt(s); });
  }
  PrimExpr VisitExpr(const PrimExpr& expr) final {
    return MutateInternal<PrimExpr>(expr,
                                    [this](const PrimExpr& e) { return this->BaseVisitExpr(e); });
  }

 private:
  // NOTE: redirect to parent's call
  // This is used to get around limitation of gcc-4.8
  Stmt BaseVisitStmt(const Stmt& s) { return StmtMutator::VisitStmt(s); }
  PrimExpr BaseVisitExpr(const PrimExpr& e) { return ExprMutator::VisitExpr(e); }

  template <typename T, typename F>
  T MutateInternal(const T& node, F fmutate) {
    if (only_enable_.size() && !only_enable_.count(node->type_index())) {
      return fmutate(node);
    }
    if (f_preorder_ != nullptr) {
      T pre = f_preorder_(node).template cast<T>();
      if (pre.defined()) return pre;
    }
    T new_node = fmutate(node);
    if (f_postorder_ != nullptr) {
      T post = f_postorder_(new_node).template cast<T>();
      if (post.defined()) return post;
    }
    return new_node;
  }
  // The functions
  const ffi::Function& f_preorder_;
  const ffi::Function& f_postorder_;
  // type indices enabled.
  const std::unordered_set<uint32_t>& only_enable_;
};

Stmt IRTransform(Stmt ir_node, const ffi::Function& f_preorder, const ffi::Function& f_postorder,
                 ffi::Optional<ffi::Array<ffi::String>> only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  if (only_enable.defined()) {
    for (auto s : only_enable.value()) {
      only_type_index.insert(ffi::TypeKeyToIndex(s.c_str()));
    }
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

class IRSubstitute : public StmtExprMutator {
 public:
  explicit IRSubstitute(std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) : vmap_(vmap) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto ret = vmap_(var);
    if (ret.defined()) {
      // Allow substitution of void variables with any expression. The TVM script parser
      // uses void variables for lambda parameters (since exact types are not known yet).
      if (!var.dtype().is_void()) {
        PrimExpr ret_ex = Downcast<PrimExpr>(ret.value());
        TVM_FFI_ICHECK(ret_ex.dtype() == var.dtype())
            << "substituting " << var << ":" << var.dtype() << " -> " << ret_ex << ":"
            << ret_ex.dtype();
      }
      return ret.value();
    }
    return var;
  }

  // Override VisitBufferDef to also remap buffer->data (the backing allocation var).
  // The base class only visits shape/strides/elem_offset.
  Buffer VisitBufferDef(const Buffer& buffer, bool alloc_data) final {
    Buffer new_buf = StmtExprMutator::VisitBufferDef(buffer, alloc_data);
    // Additionally handle data var substitution (base does not visit data).
    PrimExpr new_data_expr = VisitExpr(new_buf->data);
    TVM_FFI_ICHECK(new_data_expr->IsInstance<VarNode>())
        << "Buffer " << new_buf << " uses backing allocation " << new_buf->data
        << ", which was substituted into the expression " << new_data_expr
        << " and the backing allocation must be a tirx::Var";
    Var data = Downcast<Var>(new_data_expr);
    if (!data.same_as(new_buf->data)) {
      auto* n = new_buf.CopyOnWrite();
      n->data = std::move(data);
      buffer_remap_.Set(buffer, new_buf);
    }
    return new_buf;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (auto var_node = op->node.as<Var>()) {
      if (auto mapped_var = vmap_(var_node.value())) {
        return AttrStmt(mapped_var, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

 private:
  // Caller provided function that defines the variables to be remapped.
  std::function<ffi::Optional<PrimExpr>(const Var&)> vmap_;
};

Stmt Substitute(Stmt stmt, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstitute(vmap)(std::move(stmt));
}

PrimExpr Substitute(PrimExpr expr, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstitute(vmap)(std::move(expr));
}

void PreOrderVisit(const ObjectRef& stmt_or_expr,
                   const std::function<bool(const ObjectRef&)>& fvisit) {
  class PreOrderVisitor : public StmtExprVisitor {
   public:
    explicit PreOrderVisitor(const std::function<bool(const ObjectRef&)>& f) : f_(f) {}

   private:
    void VisitExpr(const PrimExpr& expr) final {
      const PrimExprNode* p_expr = expr.get();
      if (visited_.count(p_expr) == 0) {
        visited_.insert(p_expr);
        if (f_(expr)) {
          ExprVisitor::VisitExpr(expr);
        }
      }
    }

    void VisitStmt(const Stmt& stmt) final {
      const StmtNode* p_stmt = stmt.get();
      if (visited_.count(p_stmt) == 0) {
        visited_.insert(p_stmt);
        if (f_(stmt)) {
          StmtVisitor::VisitStmt(stmt);
        }
      }
    }

    const std::function<bool(const ObjectRef&)>& f_;
    std::unordered_set<const Object*> visited_;
  };

  PreOrderVisitor visitor(fvisit);
  if (auto stmt = stmt_or_expr.as<Stmt>()) {
    visitor(stmt.value());
  } else if (auto expr = stmt_or_expr.as<PrimExpr>()) {
    visitor(expr.value());
  } else {
    TVM_FFI_THROW(InternalError) << "PreOrderVisit does not accept object with type: "
                                 << stmt_or_expr->GetTypeKey();
  }
}

class IRSubstituteWithDataTypeLegalization : public DataTypeLegalizer {
 public:
  explicit IRSubstituteWithDataTypeLegalization(
      std::function<ffi::Optional<PrimExpr>(const Var&)> vmap)
      : vmap_(vmap) {}

  using DataTypeLegalizer::VisitExpr_;
  using DataTypeLegalizer::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto ret = vmap_(var);
    if (ret.defined()) {
      return ret.value();
    }
    return var;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (auto var_node = op->node.as<Var>()) {
      if (auto mapped_var = vmap_(var_node.value())) {
        return AttrStmt(mapped_var, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

 private:
  // Caller provided function that defines the variables to be remapped.
  std::function<ffi::Optional<PrimExpr>(const Var&)> vmap_;
};

Stmt SubstituteWithDataTypeLegalization(Stmt stmt,
                                        std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstituteWithDataTypeLegalization(vmap)(std::move(stmt));
}

PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstituteWithDataTypeLegalization(vmap)(std::move(expr));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.IRTransform", IRTransform)
      .def("tirx.PostOrderVisit",
           [](ObjectRef node, ffi::Function f) {
             tirx::PostOrderVisit(node, [f](const ObjectRef& n) { f(n); });
           })
      .def("tirx.PreOrderVisit",
           [](ObjectRef node, ffi::Function f) {
             tirx::PreOrderVisit(node, [f](const ObjectRef& n) { return f(n).cast<bool>(); });
           })
      .def("tirx.Substitute", [](ObjectRef node, ffi::Map<Var, PrimExpr> vmap) -> ObjectRef {
        if (node->IsInstance<StmtNode>()) {
          return Substitute(Downcast<Stmt>(node), vmap);
        } else {
          return Substitute(Downcast<PrimExpr>(node), vmap);
        }
      });
}

}  // namespace tirx
}  // namespace tvm
