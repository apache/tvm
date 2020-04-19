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
#include <tvm/tir/stmt_functor.h>
#include "functor_common.h"

namespace tvm {
namespace tir {

// visitor to implement apply
class IRApplyVisit :
      public StmtExprVisitor {
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

void PostOrderVisit(const ObjectRef& node,
                    std::function<void(const ObjectRef&)> fvisit) {
  if (node.as<StmtNode>()) {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<Stmt>(node));
  } else {
    IRApplyVisit visitor(fvisit);
    visitor(Downcast<PrimExpr>(node));
  }
}

class IRTransformer final :
      public StmtExprMutator {
 public:
  IRTransformer(const runtime::PackedFunc& f_preorder,
                const runtime::PackedFunc& f_postorder,
                const std::unordered_set<uint32_t>& only_enable)
      : f_preorder_(f_preorder),
        f_postorder_(f_postorder),
        only_enable_(only_enable) {
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    return MutateInternal<Stmt>(stmt, [this](const Stmt& s) {
      return this->BaseVisitStmt(s);
    });
  }
  PrimExpr VisitExpr(const PrimExpr& expr) final {
    return MutateInternal<PrimExpr>(expr, [this](const PrimExpr& e) {
      return this->BaseVisitExpr(e);
    });
  }

 private:
  // NOTE: redirect to parent's call
  // This is used to get around limitation of gcc-4.8
  Stmt BaseVisitStmt(const Stmt& s) {
    return StmtMutator::VisitStmt(s);
  }
  PrimExpr BaseVisitExpr(const PrimExpr& e) {
    return ExprMutator::VisitExpr(e);
  }

  template <typename T, typename F>
  T MutateInternal(const T& node, F fmutate) {
    if (only_enable_.size() &&
        !only_enable_.count(node->type_index())) {
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

Stmt IRTransform(Stmt ir_node,
                 const runtime::PackedFunc& f_preorder,
                 const runtime::PackedFunc& f_postorder,
                 const Array<runtime::String>& only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  for (auto s : only_enable) {
    only_type_index.insert(Object::TypeKey2Index(s.c_str()));
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

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

void StmtVisitor::VisitStmt_(const AllocateNode* op) {
  VisitArray(op->extents, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitStmt(op->body);
  this->VisitExpr(op->condition);
}

void StmtVisitor::VisitStmt_(const StoreNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
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

void StmtVisitor::VisitStmt_(const FreeNode* op) {}

void StmtVisitor::VisitStmt_(const AssertStmtNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->message);
  this->VisitStmt(op->body);
}

void StmtVisitor::VisitStmt_(const ProvideNode* op) {
  VisitArray(op->args, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->value);
}

void StmtVisitor::VisitStmt_(const RealizeNode* op) {
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
  VisitArray(op->seq, [this](const Stmt& s) {
      this->VisitStmt(s);
    });
}

void StmtVisitor::VisitStmt_(const EvaluateNode* op) {
  this->VisitExpr(op->value);
}


class StmtMutator::Internal {
 public:
  static Array<PrimExpr> Mutate(StmtMutator* self, const Array<PrimExpr>& arr) {
    auto fmutate = [self](const PrimExpr& e) { return self->VisitExpr(e); };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }

  static Array<Stmt> Mutate(StmtMutator* self, const Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }

  static Array<Range> Mutate(StmtMutator* self, const Array<Range>& arr) {
    auto fmutate = [self](const Range& r) {
      PrimExpr min = self->VisitExpr(r->min);
      PrimExpr extent = self->VisitExpr(r->extent);
      if (min.same_as(r->min) && extent.same_as(r->extent)) {
        return r;
      } else {
        return Range::make_by_min_extent(min, extent);
      }
    };
    return MutateArray(arr, fmutate, self->allow_copy_on_write_);
  }
};

Stmt StmtMutator::VisitStmt_(const AttrStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
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
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
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
  if (min.same_as(op->min) &&
      extent.same_as(op->extent) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AllocateNode* op) {
  Array<PrimExpr> extents = Internal::Mutate(this, op->extents);
  Stmt body = this->VisitStmt(op->body);
  PrimExpr condition = this->VisitExpr(op->condition);

  if (extents.same_as(op->extents) &&
      body.same_as(op->body) &&
      condition.same_as(op->condition)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->extents = std::move(extents);
    n->body = std::move(body);
    n->condition = std::move(condition);
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
  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
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
  PrimExpr value = this->VisitExpr(op->value);
  PrimExpr index = this->VisitExpr(op->index);
  PrimExpr predicate = this->VisitExpr(op->predicate);
  if (value.same_as(op->value) &&
      index.same_as(op->index) &&
      predicate.same_as(op->predicate)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->index = std::move(index);
    n->predicate = std::move(predicate);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BufferStoreNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  Array<PrimExpr> indices = Internal::Mutate(this, op->indices);

  if (value.same_as(op->value) &&
      indices.same_as(op->indices)) {
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

  if (bounds.same_as(op->bounds) &&
      condition.same_as(op->condition) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->bounds = std::move(bounds);
    n->condition = std::move(condition);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ProvideNode* op) {
  Array<PrimExpr> args = Internal::Mutate(this, op->args);
  PrimExpr value = this->VisitExpr(op->value);
  if (args.same_as(op->args) &&
      value.same_as(op->value)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->args = std::move(args);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const RealizeNode* op) {
  Region bounds = Internal::Mutate(this, op->bounds);
  Stmt body = this->VisitStmt(op->body);
  PrimExpr condition = this->VisitExpr(op->condition);
  if (bounds.same_as(op->bounds) &&
      body.same_as(op->body) &&
      condition.same_as(op->condition)) {
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
Stmt StmtMutator::VisitSeqStmt_(const SeqStmtNode* op,
                                bool flatten_before_visit,
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
    Array<Stmt> seq =
        fmutate != nullptr ?
        MutateArray(op->seq, fmutate, allow_copy_on_write_) :
        Internal::Mutate(this, op->seq);
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

  if (condition.same_as(op->condition) &&
      message.same_as(op->message) &&
      body.same_as(op->body)) {
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

Stmt StmtMutator::VisitStmt_(const FreeNode* op) {
  return GetRef<Stmt>(op);
}



}  // namespace tir
}  // namespace tvm
