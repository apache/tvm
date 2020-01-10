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
 * \file ir_functor.cc
 */
#include <tvm/ir_functor_ext.h>
#include <tvm/packed_func_ext.h>

namespace tvm {
namespace ir {

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
                 const Array<PrimExpr>& only_enable) {
  std::unordered_set<uint32_t> only_type_index;
  for (PrimExpr s : only_enable) {
    only_type_index.insert(Object::TypeKey2Index(s.as<StringImmNode>()->value.c_str()));
  }
  IRTransformer transform(f_preorder, f_postorder, only_type_index);
  return transform(std::move(ir_node));
}

// Implementation of Visitors
template<typename T, typename F>
inline void VisitArray(const Array<T>& arr, F fvisit) {
  for (size_t i = 0; i < arr.size(); i++) {
    fvisit(arr[i]);
  }
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
  if (op->new_expr.defined()) {
    this->VisitExpr(op->new_expr);
  }
}

void StmtVisitor::VisitStmt_(const StoreNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
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

void StmtVisitor::VisitStmt_(const ProducerConsumerNode* op) {
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

void ExprVisitor::VisitExpr_(const VarNode* op) {}

void ExprVisitor::VisitExpr_(const LoadNode* op) {
  this->VisitExpr(op->index);
  this->VisitExpr(op->predicate);
}

void ExprVisitor::VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  VisitArray(op->args, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

#define DEFINE_BINOP_VISIT_(OP)                           \
  void ExprVisitor::VisitExpr_(const OP* op) {            \
    this->VisitExpr(op->a);                               \
    this->VisitExpr(op->b);                               \
  }

DEFINE_BINOP_VISIT_(AddNode);
DEFINE_BINOP_VISIT_(SubNode);
DEFINE_BINOP_VISIT_(MulNode);
DEFINE_BINOP_VISIT_(DivNode);
DEFINE_BINOP_VISIT_(ModNode);
DEFINE_BINOP_VISIT_(FloorDivNode);
DEFINE_BINOP_VISIT_(FloorModNode);
DEFINE_BINOP_VISIT_(MinNode);
DEFINE_BINOP_VISIT_(MaxNode);
DEFINE_BINOP_VISIT_(EQNode);
DEFINE_BINOP_VISIT_(NENode);
DEFINE_BINOP_VISIT_(LTNode);
DEFINE_BINOP_VISIT_(LENode);
DEFINE_BINOP_VISIT_(GTNode);
DEFINE_BINOP_VISIT_(GENode);
DEFINE_BINOP_VISIT_(AndNode);
DEFINE_BINOP_VISIT_(OrNode);

void ExprVisitor::VisitExpr_(const IntImmNode* op) {}
void ExprVisitor::VisitExpr_(const UIntImmNode* op) {}
void ExprVisitor::VisitExpr_(const FloatImmNode* op) {}
void ExprVisitor::VisitExpr_(const StringImmNode* op) {}

void ExprVisitor::VisitExpr_(const ReduceNode* op) {
  VisitArray(op->axis, [this](const IterVar& r) {
      this->VisitExpr(r->dom->min);
      this->VisitExpr(r->dom->extent);
    });
  VisitArray(op->source, [this](const PrimExpr& e) { this->VisitExpr(e); });
  this->VisitExpr(op->condition);
}

void ExprVisitor::VisitExpr_(const CastNode* op) {
  this->VisitExpr(op->value);
}

void ExprVisitor::VisitExpr_(const NotNode* op) {
  this->VisitExpr(op->a);
}

void ExprVisitor::VisitExpr_(const SelectNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
}

void ExprVisitor::VisitExpr_(const RampNode* op) {
  this->VisitExpr(op->base);
  this->VisitExpr(op->stride);
}

void ExprVisitor::VisitExpr_(const ShuffleNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
  VisitArray(op->vectors, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const BroadcastNode* op) {
  this->VisitExpr(op->value);
}

// Implementation of mutators
template<typename T, typename F>
inline Array<T> MutateArray(const Array<T>& arr,
                            F fmutate,
                            bool allow_copy_on_write = false) {
  if (allow_copy_on_write) {
    // if we allow copy on write, we can directly
    // call the inplace mutate function.
    const_cast<Array<T>&>(arr).MutateByApply(fmutate);
    return arr;
  } else {
    Array<T> copy = arr;
    copy.MutateByApply(fmutate);
    return copy;
  }
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
  PrimExpr new_expr;
  if (op->new_expr.defined()) {
    new_expr = this->VisitExpr(op->new_expr);
  }
  if (extents.same_as(op->extents) &&
      body.same_as(op->body) &&
      condition.same_as(op->condition) &&
      new_expr.same_as(op->new_expr)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->extents = std::move(extents);
    n->body = std::move(body);
    n->condition = std::move(condition);
    n->new_expr = std::move(new_expr);
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

Stmt StmtMutator::VisitStmt_(const ProducerConsumerNode* op) {
  Stmt body = this->VisitStmt(op->body);
  if (body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
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


PrimExpr ExprMutator::VisitExpr_(const VarNode* op) {
  return GetRef<PrimExpr>(op);
}

PrimExpr ExprMutator::VisitExpr_(const LoadNode* op) {
  PrimExpr index = this->VisitExpr(op->index);
  PrimExpr predicate = this->VisitExpr(op->predicate);
  if (index.same_as(op->index) && predicate.same_as(op->predicate)) {
    return GetRef<PrimExpr>(op);
  } else {
    return LoadNode::make(op->dtype, op->buffer_var, index, predicate);
  }
}

PrimExpr ExprMutator::VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return LetNode::make(op->var, value, body);
  }
}

PrimExpr ExprMutator::VisitExpr_(const CallNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> args = MutateArray(op->args, fmutate);

  if (args.same_as(op->args)) {
    return GetRef<PrimExpr>(op);
  } else {
    return CallNode::make(op->dtype,
                      op->name,
                      args,
                      op->call_type,
                      op->func,
                      op->value_index);
  }
}

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP)                    \
  PrimExpr ExprMutator::VisitExpr_(const OP *op) {                    \
    return GetRef<PrimExpr>(op);                                      \
  }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(UIntImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(StringImmNode)

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                                    \
  PrimExpr ExprMutator::VisitExpr_(const OP* op) {                          \
    PrimExpr a = this->VisitExpr(op->a);                                    \
    PrimExpr b = this->VisitExpr(op->b);                                    \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
      return GetRef<PrimExpr>(op);                                          \
    } else {                                                            \
      return OP::make(a, b);                                            \
    }                                                                   \
  }

DEFINE_BIOP_EXPR_MUTATE_(AddNode);
DEFINE_BIOP_EXPR_MUTATE_(SubNode);
DEFINE_BIOP_EXPR_MUTATE_(MulNode);
DEFINE_BIOP_EXPR_MUTATE_(DivNode);
DEFINE_BIOP_EXPR_MUTATE_(ModNode);
DEFINE_BIOP_EXPR_MUTATE_(FloorDivNode);
DEFINE_BIOP_EXPR_MUTATE_(FloorModNode);
DEFINE_BIOP_EXPR_MUTATE_(MinNode);
DEFINE_BIOP_EXPR_MUTATE_(MaxNode);
DEFINE_BIOP_EXPR_MUTATE_(EQNode);
DEFINE_BIOP_EXPR_MUTATE_(NENode);
DEFINE_BIOP_EXPR_MUTATE_(LTNode);
DEFINE_BIOP_EXPR_MUTATE_(LENode);
DEFINE_BIOP_EXPR_MUTATE_(GTNode);
DEFINE_BIOP_EXPR_MUTATE_(GENode);
DEFINE_BIOP_EXPR_MUTATE_(AndNode);
DEFINE_BIOP_EXPR_MUTATE_(OrNode);

PrimExpr ExprMutator::VisitExpr_(const ReduceNode* op) {
  auto fitervar =  [this](const IterVar& v) {
    Range r = v->dom;
    PrimExpr min = this->VisitExpr(r->min);
    PrimExpr extent = this->VisitExpr(r->extent);
    if (min.same_as(r->min) &&
        extent.same_as(r->extent)) {
      return v;
    } else {
      return IterVarNode::make(
          Range::make_by_min_extent(min, extent),
          v->var, v->iter_type, v->thread_tag);
    }
  };
  Array<IterVar> axis = MutateArray(op->axis, fitervar);

  auto fexpr = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> source = MutateArray(op->source, fexpr);

  PrimExpr condition = this->VisitExpr(op->condition);

  if (axis.same_as(op->axis) &&
      source.same_as(op->source) &&
      condition.same_as(op->condition)) {
    return GetRef<PrimExpr>(op);
  } else {
    return ReduceNode::make(
      op->combiner, source, axis, condition, op->value_index);
  }
}

PrimExpr ExprMutator::VisitExpr_(const CastNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return CastNode::make(op->dtype, value);
  }
}

PrimExpr ExprMutator::VisitExpr_(const NotNode* op) {
  PrimExpr a = this->VisitExpr(op->a);
  if (a.same_as(op->a)) {
    return GetRef<PrimExpr>(op);
  } else {
    return NotNode::make(a);
  }
}

PrimExpr ExprMutator::VisitExpr_(const SelectNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr true_value = this->VisitExpr(op->true_value);
  PrimExpr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) &&
      true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return SelectNode::make(condition, true_value, false_value);
  }
}

PrimExpr ExprMutator::VisitExpr_(const RampNode* op) {
  PrimExpr base = this->VisitExpr(op->base);
  PrimExpr stride = this->VisitExpr(op->stride);
  if (base.same_as(op->base) &&
      stride.same_as(op->stride)) {
    return GetRef<PrimExpr>(op);
  } else {
    return RampNode::make(base, stride, op->lanes);
  }
}

PrimExpr ExprMutator::VisitExpr_(const BroadcastNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return BroadcastNode::make(value, op->lanes);
  }
}

PrimExpr ExprMutator::VisitExpr_(const ShuffleNode* op) {
  auto fexpr = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  auto vectors = MutateArray(op->vectors, fexpr);
  if (vectors.same_as(op->vectors)) {
    return GetRef<PrimExpr>(op);
  } else {
    return ShuffleNode::make(vectors, op->indices);
  }
}

}  // namespace ir
}  // namespace tvm
