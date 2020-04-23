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
 * \file narrow_datatype.cc
 * \brief narrow the datatype of indexing vars
 */

#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/registry.h>
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

// This pass narrows indexing expressions (like StoreNode::Index)
// that trivially fit into i32/i16 (denoted by `target_bits_`) to
// i32/i16. Considering that i32/i16 indices may be more
// efficient on some backends (while i64 may be more efficient
// on others, like llvm), we may want this pass when i32/i16
// indices are more efficient.
//
// For Var v, we determine its dtype by examining all the PrimExpr
// that contains v, denoted by E = {e_0 = v, e_1, e_2, ..., e_k}.
// If all expressions in E fit into i32/i16, then we think v can be narrowed
// to i32/i16.
//
// To make an indexing expression i32/i16, we must make sure that every
// component of that expression is of dtype i32/i16. So besides Var, we
// rewrite the following inside an indexing expression
// - Var
// - IntImm
// - Cast
//
// Algorithm:
// - Use DataTypeVisitor to determine whether a Var can be narrowed or not.
// - Use DataTypeRewritter to rewrite the components of an indexing expression.

using arith::Analyzer;
using arith::IRMutatorWithAnalyzer;
using arith::ConstIntBound;

// Determine the result dtype for Var, IntImm and Cast,
// which will be stored in `vmap` eventually.
//
// Algorithm:
// We propogate the dtypes of all the Exprs that contain Var `var` into `vmap[var]`.
// To be more specific, if for each Expr `e` which contains `var`
// (`var` is a child node of `e` in AST), `e` fits into `target_bits_`,
// then we narrow `var` into `target_bits_`. That is,
// `vmap[var] = min(target_bits_, var.dtype.bits())`
// Otherwise, `var` is not narrowed, that is, `vmap[var] = var.dtype.bits()`
class DataTypeVisitor final : public StmtExprVisitor {
 public:
  explicit DataTypeVisitor(int target_bits)
    : bits_(target_bits), target_bits_(target_bits) {}

  void VisitExpr(const PrimExpr& e) {
    if (e.dtype().is_int()) {
      int bits = max_bits_;
      const PrimExprNode* op = e.as<PrimExprNode>();
      if (bound_.find(op) == bound_.end()) {
        analyzer_.const_int_bound(e, &bound_);
      }
      ConstIntBound bound = bound_[op];
      int64_t ubound = Downcast<IntImm>(max_value(DataType::Int(target_bits_)))->value;
      int64_t lbound = Downcast<IntImm>(min_value(DataType::Int(target_bits_)))->value;
      if (e.dtype().bits() <= target_bits_ ||
          (bound->max_value <= ubound && bound->min_value >= lbound)) {
        bits = target_bits_;
      }
      int tmp = bits > bits_ ? bits :  bits_;
      std::swap(bits_, tmp);
      StmtExprVisitor::VisitExpr(e);
      std::swap(bits_, tmp);
    } else {
      StmtExprVisitor::VisitExpr(e);
    }
  }

  void VisitStmt_(const ForNode* op) {
    analyzer_.Bind(op->loop_var,
                   Range::make_by_min_extent(op->min, op->extent));
    vextent_[op->loop_var.as<VarNode>()] = op->extent.dtype();
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK_NE(iv->thread_tag.length(), 0U);
      analyzer_.Bind(iv->var,
                      Range::make_by_min_extent(0, op->value));
      vextent_[iv->var.as<VarNode>()] = op->value.dtype();
      StmtExprVisitor::VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const ReduceNode* op) {
    // Setup the domain information before simplification.
    for (const IterVar& iv : op->axis) {
      analyzer_.Bind(iv->var, iv->dom);
      vextent_[iv->var.as<VarNode>()] = iv->dom->extent.dtype();
    }
    // Recursively call simplification when necessary.
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) {
    if (vextent_.find(op) != vextent_.end()) {
      // We only narrow and never promote, so the result dtype
      // is upperbounded by its original dtype before rewrite.
      int bits = std::min(vextent_[op].bits(), bits_);
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits);
      } else {
        // We take maximum bits for all the possible Expr where a var occurs
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IntImmNode* op) {
    if (op->dtype.is_int()) {
      // We only narrow and never promote, so the result dtype
      // is upperbounded by its original dtype before rewrite.
      int bits = std::min(op->dtype.bits(), bits_);
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits);
      } else {
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CastNode* op) {
    if (op->dtype.is_int()) {
      // We only narrow and never promote, so the result dtype
      // is upperbounded by its original dtype before rewrite.
      int bits = std::min(op->dtype.bits(), bits_);
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits);
      } else {
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // the narrowed datatype of Var and IntImm
  std::unordered_map<const PrimExprNode*, DataType> vmap;

 protected:
  // internal analyzer
  arith::Analyzer analyzer_;

 private:
  // the maximum possible bits, which serves as an init value
  static constexpr const int max_bits_ = 64;
  // the maximum possible bit of the current expression's return dtype
  int bits_;
  // the target bits
  int target_bits_;
  // the extent of vars to be rewritten
  std::unordered_map<const VarNode*, DataType> vextent_;
  // the memorized bound generated by ConstIntBoundAnalyzer
  std::unordered_map<const PrimExprNode*, ConstIntBound> bound_;
};

class DataTypeRewriter : public StmtExprMutator {
 public:
  explicit DataTypeRewriter(int target_bits): visitor_(target_bits) {}

  Stmt operator()(Stmt s) {
    visitor_(s);
    for (auto i = visitor_.vmap.begin(), last = visitor_.vmap.end(); i != last;) {
      PrimExpr e = GetRef<PrimExpr>(i->first);
      if (e.dtype() == i->second) {
        i = visitor_.vmap.erase(i);
      } else {
        ++i;
      }
    }
    return VisitStmt(s);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    is_index_ = true;
    PrimExpr index = this->VisitExpr(op->index);
    is_index_ = false;
    Stmt s = StoreNode::make(op->buffer_var,
                             op->value,
                             index,
                             op->predicate);
    return StmtExprMutator::VisitStmt_(s.as<StoreNode>());
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt s = StmtExprMutator::VisitStmt_(op);
    op = s.as<ForNode>();
    CHECK(op != nullptr)
      << "Expected type to be ForNode"
      << ", but get " << s->GetTypeKey();
    PrimExpr e = VisitExpr(op->loop_var);
    Var var = Downcast<Var>(e);
    return ForNode::make(var, cast(var.dtype(), op->min), cast(var.dtype(), op->extent),
                         op->for_type, op->device_api, op->body);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      Stmt s = StmtExprMutator::VisitStmt_(op);
      op = s.as<AttrStmtNode>();
      CHECK(op != nullptr)
        << "Expected type to be AttrStmtNode"
        << ", but get " << s->GetTypeKey();
      const IterVarNode* iv = op->node.as<IterVarNode>();
      CHECK(iv != nullptr)
        << "Expected type to be IterVarNode"
        << ", but get " << op->node->GetTypeKey();
      PrimExpr e = VisitExpr(iv->var);
      Var var = Downcast<Var>(e);
      if (ivmap_.find(iv) == ivmap_.end()) {
        ivmap_[iv] = IterVarNode::make(iv->dom, var, iv->iter_type, iv->thread_tag);
      }
      return AttrStmtNode::make(
        ivmap_[iv],
        op->attr_key,
        cast(var.dtype(), op->value),
        op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = Var(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const SizeVarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = SizeVar(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    is_index_ = true;
    PrimExpr index = this->VisitExpr(op->index);
    is_index_ = false;
    PrimExpr e = LoadNode::make(op->dtype, op->buffer_var, index, op->predicate);
    return StmtExprMutator::VisitExpr_(e.as<LoadNode>());
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    if (is_index_) {
      if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
        return IntImm(visitor_.vmap[op], op->value);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    if (is_index_ && visitor_.vmap.find(op) != visitor_.vmap.end()) {
      PrimExpr e = StmtExprMutator::VisitExpr_(op);
      const CastNode* new_op = e.as<CastNode>();
      CHECK(new_op != nullptr)
        << "Expected type to be CastNode"
        << ", but get " << e->GetTypeKey();
      return CastNode::make(visitor_.vmap[op], new_op->value);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const MinNode* op) final;
  PrimExpr VisitExpr_(const MaxNode* op) final;
  PrimExpr VisitExpr_(const EQNode* op) final;
  PrimExpr VisitExpr_(const NENode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;
  PrimExpr VisitExpr_(const LENode* op) final;
  PrimExpr VisitExpr_(const GTNode* op) final;
  PrimExpr VisitExpr_(const GENode* op) final;
  PrimExpr VisitExpr_(const CallNode* op) final;

 private:
  // the internal visitor to deduce the narrowed dtype
  DataTypeVisitor visitor_;
  // a map from Var before rewrite to that after rewrite,
  // ensures one old Var maps to exactly one new Var
  std::unordered_map<const VarNode*, Var> vmap_;
  // a map from IterVar before rewrite to that after rewrite,
  // ensures one old IterVar maps to exactly one new IterVar
  std::unordered_map<const IterVarNode*, IterVar> ivmap_;
  // indicator of LoadNode::index and StoreNode::index
  bool is_index_{false};
};

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)               \
  PrimExpr DataTypeRewriter::VisitExpr_(const OP* op) {                 \
    PrimExpr a = this->VisitExpr(op->a);                                \
    PrimExpr b = this->VisitExpr(op->b);                                \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
      return GetRef<PrimExpr>(op);                                      \
    } else {                                                            \
      return FUNC(a, b);                                                \
    }                                                                   \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator <)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator >)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=)

PrimExpr DataTypeRewriter::VisitExpr_(const CallNode* op) {
  PrimExpr e = StmtExprMutator::VisitExpr_(op);
  op = e.as<CallNode>();
  CHECK(op != nullptr)
    << "Expected type to be CallNode"
    << ", but get " << e->GetTypeKey();
  if (op->call_type == CallNode::PureIntrinsic) {
    if (op->name == intrinsic::tvm_if_then_else) {
      return if_then_else(op->args[0], op->args[1], op->args[2]);
    } else if (op->name == CallNode::shift_right) {
      return op->args[0] >> op->args[1];
    } else if (op->name == CallNode::shift_left) {
      return op->args[0] << op->args[1];
    } else if (op->name == CallNode::bitwise_and) {
      return op->args[0] & op->args[1];
    } else if (op->name == CallNode::bitwise_or) {
      return op->args[0] | op->args[1];
    } else if (op->name == CallNode::bitwise_xor) {
      return op->args[0] ^ op->args[1];
    } else if (op->name == "pow") {
      return pow(op->args[0], op->args[1]);
    }
  }
  return e;
}

Stmt NarrowDataType(Stmt stmt, int target_bits) {
  return DataTypeRewriter(target_bits)(stmt);
}

namespace transform {

Pass NarrowDataType(int target_bits) {
  auto pass_func = [target_bits](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = DataTypeRewriter(target_bits)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(
      pass_func, 0, "tir.NarrowDataType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.NarrowDataType")
.set_body_typed(NarrowDataType);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
