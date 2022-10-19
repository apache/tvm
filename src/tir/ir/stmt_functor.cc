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
#include <tvm/ir/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>

#include "functor_common.h"

namespace tvm {
namespace tir {

void StmtVisitor::VisitStmt_(const LetStmtNode* op) {
  this->VisitExpr(op->value);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const AttrStmtNode* op) {
  this->VisitExpr(op->value);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const ForNode* op) {
  this->VisitExpr(op->min);
  this->VisitExpr(op->extent);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const WhileNode* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const AllocateNode* op) {
  VisitArray(op->extents, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitStmt(op->body);
  this->VisitExpr(op->condition);
}

void StmtVisitor::VisitStmt_(const AllocateConstNode* op) {
  VisitArray(op->extents, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const DeclBufferNode* op) { this->VisitStmt(op->body); }

void StmtVisitor::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
}

void StmtVisitor::VisitStmt_(const BufferStoreNode* op) {
  this->VisitExpr(op->value);
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void StmtVisitor::VisitStmt_(const BufferRealizeNode* op) {
  VisitArray(op->bounds, [this](const Range& r) {
    this->VisitExpr(r->min);
    this->VisitExpr(r->extent);
  });
  this->VisitExpr(op->condition);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const IfThenElseNode* op) {
  this->VisitExpr(op->condition);
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    this->VisitStmt(op->else_case);
  }
}

void StmtVisitor::VisitStmt_(const AssertStmtNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->message);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const ProducerStoreNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->value);
}

void StmtVisitor::VisitStmt_(const ProducerRealizeNode* op) {
  VisitArray(op->bounds, [this](const Range& r) {
    this->VisitExpr(r->min);
    this->VisitExpr(r->extent);
  });
  this->VisitStmt(op->body);
  this->VisitExpr(op->condition);
}

void StmtVisitor::VisitStmt_(const PrefetchNode* op) {
  VisitArray(op->bounds, [this](const Range& r) {
    this->VisitExpr(r->min);
    this->VisitExpr(r->extent);
  });
}

void StmtVisitor::VisitStmt_(const SeqStmtNode* op) {
  VisitArray(op->seq, [this](const Stmt& s) { this->VisitStmt(s); });
}

void StmtVisitor::VisitStmt_(const EvaluateNode* op) { this->VisitExpr(op->value); }

void StmtVisitor::VisitStmt_(const BlockNode* op) {
  auto fvisit_buffer_region = [this](const BufferRegion& s) {
    for (const auto& range : s->region) {
      this->VisitExpr(range->min);
      this->VisitExpr(range->extent);
    }
  };
  VisitArray(op->iter_vars, [this](const IterVar& iter_var) {
    this->VisitExpr(iter_var->dom->min);
    this->VisitExpr(iter_var->dom->extent);
  });
  VisitArray(op->reads, fvisit_buffer_region);
  VisitArray(op->writes, fvisit_buffer_region);
  VisitArray(op->match_buffers,
             [fvisit_buffer_region](const MatchBufferRegion& match_buffer_region) {
               fvisit_buffer_region(match_buffer_region->source);
             });
  if (op->init.defined()) {
    this->VisitStmt(op->init.value());
  }
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const BlockRealizeNode* op) {
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
  static Array<T> MutateArray(StmtMutator* self, const Array<T>& arr, F fmutate) {
    if (self->allow_copy_on_write_ && arr.unique()) {
      // if we allow copy on write, we can directly
      // call the inplace mutate function.
      const_cast<Array<T>&>(arr).MutateByApply(fmutate);
      return arr;
    } else {
      bool allow_cow = false;
      std::swap(allow_cow, self->allow_copy_on_write_);
      Array<T> copy = arr.Map(fmutate);
      std::swap(allow_cow, self->allow_copy_on_write_);
      return copy;
    }
  }

  static Array<IterVar> Mutate(StmtMutator* self, const Array<IterVar>& arr) {
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

  static Array<PrimExpr> Mutate(StmtMutator* self, const Array<PrimExpr>& arr) {
    auto fmutate = [self](const PrimExpr& e) { return self->VisitExpr(e); };
    return MutateArray(self, arr, fmutate);
  }

  static Array<Stmt> Mutate(StmtMutator* self, const Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(self, arr, fmutate);
  }

  static Array<Range> Mutate(StmtMutator* self, const Array<Range>& arr) {
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

  static Array<BufferRegion> Mutate(StmtMutator* self, const Array<BufferRegion>& arr) {
    auto fmutate = [self](const BufferRegion& buffer_region) {
      Array<Range> region = Mutate(self, buffer_region->region);
      if (region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(buffer_region->buffer, region);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static Array<MatchBufferRegion> Mutate(StmtMutator* self, const Array<MatchBufferRegion>& arr) {
    auto fmutate = [self](const MatchBufferRegion& match_buffer_region) {
      Array<Range> region = Mutate(self, match_buffer_region->source->region);
      if (region.same_as(match_buffer_region->source->region)) {
        return match_buffer_region;
      } else {
        return MatchBufferRegion(match_buffer_region->buffer,
                                 BufferRegion(match_buffer_region->source->buffer, region));
      }
    };
    return MutateArray(self, arr, fmutate);
  }
};

Stmt StmtMutator::VisitStmt_(const AttrStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const LetStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
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
  Stmt body = this->VisitStmt(op->body);
  if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const WhileNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt body = this->VisitStmt(op->body);
  if (condition.same_as(op->condition) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AllocateNode* op) {
  Array<PrimExpr> extents = Internal::Mutate(this, op->extents);
  Stmt body = this->VisitStmt(op->body);
  PrimExpr condition = this->VisitExpr(op->condition);

  if (extents.same_as(op->extents) && body.same_as(op->body) && condition.same_as(op->condition)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->extents = std::move(extents);
    n->body = std::move(body);
    n->condition = std::move(condition);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AllocateConstNode* op) {
  Array<PrimExpr> extents = Internal::Mutate(this, op->extents);
  Stmt body = this->VisitStmt(op->body);

  if (extents.same_as(op->extents) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->extents = std::move(extents);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const DeclBufferNode* op) {
  Stmt body = this->VisitStmt(op->body);

  if (body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt then_case = this->VisitStmt(op->then_case);
  Stmt else_case;
  if (op->else_case.defined()) {
    else_case = this->VisitStmt(op->else_case);
  }
  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
  return Stmt();
}

Stmt StmtMutator::VisitStmt_(const BufferStoreNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Array<PrimExpr> indices = Internal::Mutate(this, op->indices);

  if (value.same_as(op->value) && indices.same_as(op->indices)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->indices = std::move(indices);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BufferRealizeNode* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt body = this->VisitStmt(op->body);

  if (bounds.same_as(op->bounds) && condition.same_as(op->condition) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    n->condition = std::move(condition);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ProducerStoreNode* op) {
  Array<PrimExpr> indices = Internal::Mutate(this, op->indices);
  PrimExpr value = this->VisitExpr(op->value);
  if (indices.same_as(op->indices) && value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->indices = std::move(indices);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ProducerRealizeNode* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  Stmt body = this->VisitStmt(op->body);
  PrimExpr condition = this->VisitExpr(op->condition);
  if (bounds.same_as(op->bounds) && body.same_as(op->body) && condition.same_as(op->condition)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    n->body = std::move(body);
    n->condition = std::move(condition);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const PrefetchNode* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  if (bounds.same_as(op->bounds)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SeqStmtNode* op) {
  Array<Stmt> seq = Internal::Mutate(this, op->seq);
  if (seq.same_as(op->seq)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->seq = std::move(seq);
    return Stmt(n);
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
    Array<Stmt> seq = fmutate != nullptr ? Internal::MutateArray(this, op->seq, fmutate)
                                         : Internal::Mutate(this, op->seq);
    if (seq.same_as(op->seq)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->seq = std::move(seq);
      return Stmt(n);
    }
  };
  if (flatten_before_visit) {
    Array<Stmt> seq;
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
  PrimExpr message = this->VisitExpr(op->message);
  Stmt body = this->VisitStmt(op->body);

  if (condition.same_as(op->condition) && message.same_as(op->message) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const EvaluateNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BlockNode* op) {
  Array<IterVar> iter_vars = Internal::Mutate(this, op->iter_vars);
  Array<BufferRegion> reads = Internal::Mutate(this, op->reads);
  Array<BufferRegion> writes = Internal::Mutate(this, op->writes);
  Array<MatchBufferRegion> match_buffers = Internal::Mutate(this, op->match_buffers);
  Optional<Stmt> init = NullOpt;
  if (op->init.defined()) {
    init = VisitStmt(op->init.value());
  }
  Stmt body = VisitStmt(op->body);
  if (iter_vars.same_as(op->iter_vars) && reads.same_as(op->reads) && writes.same_as(op->writes) &&
      body.same_as(op->body) && init.same_as(op->init) &&
      match_buffers.same_as(op->match_buffers)) {
    return GetRef<Block>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_vars = std::move(iter_vars);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);
    n->match_buffers = std::move(match_buffers);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BlockRealizeNode* op) {
  Array<PrimExpr> v = Internal::Mutate(this, op->iter_values);
  PrimExpr pred = this->VisitExpr(op->predicate);
  Stmt block = this->VisitStmt(op->block);
  if (v.same_as(op->iter_values) && pred.same_as(op->predicate) && block.same_as(op->block)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_values = std::move(v);
    n->predicate = std::move(pred);
    n->block = Downcast<Block>(block);
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
  IRTransformer(const runtime::PackedFunc& f_preorder, const runtime::PackedFunc& f_postorder,
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
      T pre = f_preorder_(node);
      if (pre.defined()) return pre;
    }
    T new_node = fmutate(node);
    if (f_postorder_ != nullptr) {
      T post = f_postorder_(new_node);
      if (post.defined()) return post;
    }
    return new_node;
  }
  // The functions
  const runtime::PackedFunc& f_preorder_;
  const runtime::PackedFunc& f_postorder_;
  // type indices enabled.
  const std::unordered_set<uint32_t>& only_enable_;
};

Stmt IRTransform(Stmt ir_node, const runtime::PackedFunc& f_preorder,
                 const runtime::PackedFunc& f_postorder, Optional<Array<String>> only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  if (only_enable.defined()) {
    for (auto s : only_enable.value()) {
      only_type_index.insert(Object::TypeKey2Index(s.c_str()));
    }
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

class IRSubstitute : public StmtExprMutator {
 public:
  explicit IRSubstitute(std::function<Optional<PrimExpr>(const Var&)> vmap) : vmap_(vmap) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto ret = vmap_(var);
    if (ret.defined()) {
      // Allow substitution of void variables with any expression. The TVM script parser
      // uses void variables for lambda parameters (since exact types are not known yet).
      if (!var.dtype().is_void()) {
        PrimExpr ret_ex = Downcast<PrimExpr>(ret.value());
        ICHECK(ret_ex.dtype() == var.dtype()) << "substituting " << var << ":" << var.dtype()
                                              << " -> " << ret_ex << ":" << ret_ex.dtype();
      }
      return ret.value();
    }
    return std::move(var);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    Buffer new_buf = GetRemappedBuffer(node->buffer);

    if (!new_buf.same_as(node->buffer)) {
      auto writer = node.CopyOnWrite();
      writer->buffer = new_buf;
    }

    return node;
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    auto key = buf.get();
    auto it = buf_remap_.find(key);
    if (it != buf_remap_.end()) {
      return it->second;
    }

    auto new_buffer_var = vmap_(buf->data);
    if (new_buffer_var.defined() && !new_buffer_var.value().same_as(buf->data)) {
      auto writer = buf.CopyOnWrite();
      writer->data = Downcast<Var>(new_buffer_var);
    }

    buf_remap_[key] = buf;
    return buf;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (const auto* var_node = op->node.as<VarNode>()) {
      if (auto mapped_var = vmap_(GetRef<Var>(var_node))) {
        return AttrStmt(mapped_var, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

 private:
  // Caller provided function that defines the variables to be remapped.
  std::function<Optional<PrimExpr>(const Var&)> vmap_;

  /* \brief Generated map to track buffers being remapped.
   *
   * If a `Var BufferNode::data` is remapped, then all buffers
   * containing that data pointer should also be remapped.  This map
   * is used to track buffer modifications, and ensure all instances
   * of a buffer are replaced by the same modified buffer object.
   */
  std::unordered_map<const BufferNode*, Buffer> buf_remap_;
};

Stmt Substitute(Stmt stmt, std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstitute(vmap)(std::move(stmt));
}

PrimExpr Substitute(PrimExpr expr, std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstitute(vmap)(std::move(expr));
}

Array<Range> Substitute(const Array<Range>& region, const Map<Var, PrimExpr>& vmap) {
  Array<Range> result;
  result.reserve(region.size());
  for (const Range& range : region) {
    PrimExpr min = Substitute(range->min, vmap);
    PrimExpr extent = Substitute(range->extent, vmap);
    result.push_back(Range::FromMinExtent(std::move(min), std::move(extent)));
  }
  return result;
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
  if (const auto* stmt = stmt_or_expr.as<StmtNode>()) {
    visitor(GetRef<Stmt>(stmt));
  } else if (const auto* expr = stmt_or_expr.as<PrimExprNode>()) {
    visitor(GetRef<PrimExpr>(expr));
  } else {
    LOG(FATAL) << "InternalError: PreOrderVisit does not accept object with type: "
               << stmt_or_expr->GetTypeKey();
  }
}

class IRSubstituteWithDataTypeLegalization : public DataTypeLegalizer {
 public:
  explicit IRSubstituteWithDataTypeLegalization(std::function<Optional<PrimExpr>(const Var&)> vmap)
      : vmap_(vmap) {}

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto ret = vmap_(var);
    if (ret.defined()) {
      return ret.value();
    }
    return std::move(var);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    Buffer new_buf = GetRemappedBuffer(node->buffer);

    if (!new_buf.same_as(node->buffer)) {
      auto writer = node.CopyOnWrite();
      writer->buffer = new_buf;
    }

    return node;
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    auto key = buf.get();
    auto it = buf_remap_.find(key);
    if (it != buf_remap_.end()) {
      return it->second;
    }

    auto new_buffer_var = vmap_(buf->data);
    if (new_buffer_var.defined() && !new_buffer_var.value().same_as(buf->data)) {
      auto writer = buf.CopyOnWrite();
      writer->data = Downcast<Var>(new_buffer_var);
    }

    buf_remap_[key] = buf;
    return buf;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (const auto* var_node = op->node.as<VarNode>()) {
      if (auto mapped_var = vmap_(GetRef<Var>(var_node))) {
        return AttrStmt(mapped_var, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

 private:
  // Caller provided function that defines the variables to be remapped.
  std::function<Optional<PrimExpr>(const Var&)> vmap_;

  /* \brief Generated map to track buffers being remapped.
   *
   * If a `Var BufferNode::data` is remapped, then all buffers
   * containing that data pointer should also be remapped.  This map
   * is used to track buffer modifications, and ensure all instances
   * of a buffer are replaced by the same modified buffer object.
   */
  std::unordered_map<const BufferNode*, Buffer> buf_remap_;
};

Stmt SubstituteWithDataTypeLegalization(Stmt stmt,
                                        std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstituteWithDataTypeLegalization(vmap)(std::move(stmt));
}

PrimExpr SubstituteWithDataTypeLegalization(PrimExpr expr,
                                            std::function<Optional<PrimExpr>(const Var&)> vmap) {
  return IRSubstituteWithDataTypeLegalization(vmap)(std::move(expr));
}

TVM_REGISTER_GLOBAL("tir.IRTransform").set_body_typed(IRTransform);

TVM_REGISTER_GLOBAL("tir.PostOrderVisit").set_body_typed([](ObjectRef node, PackedFunc f) {
  tir::PostOrderVisit(node, [f](const ObjectRef& n) { f(n); });
});

TVM_REGISTER_GLOBAL("tir.PreOrderVisit").set_body_typed([](ObjectRef node, PackedFunc f) {
  tir::PreOrderVisit(node, [f](const ObjectRef& n) { return f(n); });
});

TVM_REGISTER_GLOBAL("tir.Substitute")
    .set_body_typed([](ObjectRef node, Map<Var, PrimExpr> vmap) -> ObjectRef {
      if (node->IsInstance<StmtNode>()) {
        return Substitute(Downcast<Stmt>(node), vmap);
      } else {
        return Substitute(Downcast<PrimExpr>(node), vmap);
      }
    });

}  // namespace tir
}  // namespace tvm
