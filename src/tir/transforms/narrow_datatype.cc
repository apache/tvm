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

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

// This pass narrows indexing expressions (like BufferStoreNode::indices)
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
using arith::ConstIntBound;
using arith::IRMutatorWithAnalyzer;

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
  explicit DataTypeVisitor(int target_bits) : bits_(target_bits), target_bits_(target_bits) {}

  void VisitExpr(const PrimExpr& e) {
    if (e.dtype().is_int()) {
      int bits = max_bits_;
      if (bound_.find(e) == bound_.end()) {
        analyzer_.const_int_bound(e, &bound_);
      }
      ConstIntBound bound = bound_[e];
      int64_t ubound = Downcast<IntImm>(max_value(DataType::Int(target_bits_)))->value;
      int64_t lbound = Downcast<IntImm>(min_value(DataType::Int(target_bits_)))->value;
      if (e.dtype().bits() <= target_bits_ ||
          (bound->max_value <= ubound && bound->min_value >= lbound)) {
        bits = target_bits_;
      }
      int tmp = bits > bits_ ? bits : bits_;
      std::swap(bits_, tmp);
      StmtExprVisitor::VisitExpr(e);
      std::swap(bits_, tmp);
    } else {
      StmtExprVisitor::VisitExpr(e);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) {
    int tmp = bits_;
    bits_ = target_bits_;
    StmtExprVisitor::VisitExpr_(op);
    bits_ = tmp;
  }

  void VisitStmt_(const ForNode* op) {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    vextent_[op->loop_var.as<VarNode>()] = op->extent.dtype();
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode* op) {
    for (const IterVar& iter : op->iter_vars) {
      analyzer_.Bind(iter->var, Range::FromMinExtent(iter->dom->min, iter->dom->extent));
      vextent_[iter->var.as<VarNode>()] = iter->dom->extent.dtype();
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      analyzer_.Bind(iv->var, Range::FromMinExtent(0, op->value));
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
  arith::ConstIntBoundAnalyzer::BoundMapType bound_;
};

class NarrowDataTypeRewriter : public IndexDataTypeRewriter {
 public:
  using Parent = IndexDataTypeRewriter;
  explicit NarrowDataTypeRewriter(int target_bits) : visitor_(target_bits) {}

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

 protected:
  // This class adds some overrides of `VisitStmt_` and `VisitExpr_` that
  // are *not* present in the parent class.
  // These `using` statements ensure that all of the *other* overrides
  // provided by the parent class are fully visible to users of this class.
  // (Discussed further in https://github.com/apache/tvm/pull/13267)
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (auto it = visitor_.vmap.find(op); !var_remap_.count(op) && it != visitor_.vmap.end()) {
      var_remap_[op] = Var(op->name_hint, it->second);
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    if (is_enabled_) {
      if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
        return IntImm(visitor_.vmap[op], op->value);
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    if (is_enabled_ && visitor_.vmap.find(op) != visitor_.vmap.end()) {
      PrimExpr e = Parent::VisitExpr_(op);
      const CastNode* new_op = e.as<CastNode>();
      ICHECK(new_op != nullptr) << "Expected type to be CastNode"
                                << ", but get " << e->GetTypeKey();
      PrimExpr new_value = new_op->value;
      DataType cast_type = visitor_.vmap[op];
      if (new_value.dtype() != cast_type) {
        new_value = Cast(cast_type, new_value);
      }
      return new_value;
    }
    return Parent::VisitExpr_(op);
  }

#define TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)             \
  PrimExpr VisitExpr_(const OP* op) {                                     \
    PrimExpr a = this->VisitExpr(op->a);                                  \
    PrimExpr b = this->VisitExpr(op->b);                                  \
    if (op->a.same_as(a) && op->b.same_as(b) && a.dtype() == b.dtype()) { \
      return GetRef<PrimExpr>(op);                                        \
    } else {                                                              \
      if (a.dtype() != b.dtype()) {                                       \
        bool is_enabled = is_enabled_;                                    \
        is_enabled_ = true;                                               \
        PrimExpr lhs = this->VisitExpr(op->a);                            \
        PrimExpr rhs = this->VisitExpr(op->b);                            \
        is_enabled_ = is_enabled;                                         \
        return FUNC(lhs, rhs);                                            \
      } else {                                                            \
        return FUNC(a, b);                                                \
      }                                                                   \
    }                                                                     \
  }

  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

#undef TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

 private:
  // the internal visitor to deduce the narrowed dtype
  DataTypeVisitor visitor_;
};

Stmt NarrowDataType(Stmt stmt, int target_bits) {
  return NarrowDataTypeRewriter(target_bits)(stmt);
}

namespace transform {

Pass NarrowDataType(int target_bits) {
  auto pass_func = [target_bits](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = NarrowDataTypeRewriter(target_bits)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.NarrowDataType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.NarrowDataType").set_body_typed(NarrowDataType);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
