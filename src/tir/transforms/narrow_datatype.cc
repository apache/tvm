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
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

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

  void VisitStmt_(const ForNode* op) {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    vextent_[op->loop_var.as<VarNode>()] = op->extent.dtype();
    return StmtExprVisitor::VisitStmt_(op);
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

class DataTypeRewriter : public DataTypeLegalizer {
  using Parent = DataTypeLegalizer;

 public:
  explicit DataTypeRewriter(int target_bits) : visitor_(target_bits) {}

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
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = GetRef<BufferStore>(op);

    auto value = this->VisitExpr(op->value);
    auto indices = VisitIndices(op->indices);

    if (!value.same_as(op->value) || !indices.same_as(op->indices)) {
      auto writer = store.CopyOnWrite();
      writer->value = value;
      writer->indices = indices;
    }

    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad load = GetRef<BufferLoad>(op);

    auto indices = VisitIndices(op->indices);

    if (!indices.same_as(op->indices)) {
      auto writer = load.CopyOnWrite();
      writer->indices = indices;
    }

    return std::move(load);
  }

  Array<PrimExpr> VisitIndices(Array<PrimExpr> indices) {
    is_index_ = true;

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    indices.MutateByApply(fmutate);

    is_index_ = false;

    return indices;
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    IfThenElse updated = Downcast<IfThenElse>(Parent::VisitStmt_(op));
    is_condition_ = true;
    PrimExpr cond = VisitExpr(op->condition);
    is_condition_ = false;
    if (!cond.same_as(op->condition)) {
      return std::move(IfThenElse(cond, updated->then_case, updated->else_case));
    }
    return std::move(updated);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = Var(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const SizeVarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = SizeVar(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    if (is_index_) {
      if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
        return IntImm(visitor_.vmap[op], op->value);
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    if (is_index_ && visitor_.vmap.find(op) != visitor_.vmap.end()) {
      PrimExpr e = Parent::VisitExpr_(op);
      const CastNode* new_op = e.as<CastNode>();
      ICHECK(new_op != nullptr) << "Expected type to be CastNode"
                                << ", but get " << e->GetTypeKey();
      return Cast(visitor_.vmap[op], new_op->value);
    }
    return Parent::VisitExpr_(op);
  }

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
  // indicator of index expr to rewrite
  bool is_index_{false};
  // indicator of condition
  bool is_condition_{false};
};

#define DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                          \
  PrimExpr DataTypeRewriter::VisitExpr_(const OP* op) {                             \
    bool is_index = is_index_;                                                      \
    bool rewrite = is_condition_ && op->a->dtype.is_int() && op->b->dtype.is_int(); \
    if (rewrite) {                                                                  \
      is_index_ = true;                                                             \
    }                                                                               \
    auto result = Parent::VisitExpr_(op);                                           \
    is_index_ = is_index;                                                           \
    return std::move(result);                                                       \
  }

DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

PrimExpr DataTypeRewriter::VisitExpr_(const CallNode* op) {
  // handle if_then_else condition
  if (op->op.same_as(builtin::if_then_else())) {
    bool is_condition = is_condition_;
    is_condition_ = true;
    PrimExpr cond = VisitExpr(op->args[0]);
    is_condition_ = is_condition;
    return if_then_else(cond, VisitExpr(op->args[1]), VisitExpr(op->args[2]));
  }
  return Parent::VisitExpr_(op);
}

Stmt NarrowDataType(Stmt stmt, int target_bits) { return DataTypeRewriter(target_bits)(stmt); }

namespace transform {

Pass NarrowDataType(int target_bits) {
  auto pass_func = [target_bits](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = DataTypeRewriter(target_bits)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.NarrowDataType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.NarrowDataType").set_body_typed(NarrowDataType);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
