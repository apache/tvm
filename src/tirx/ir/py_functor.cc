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
 * \file src/tirx/ir/py_functor.cc
 * \brief The python interface of ExprVisitor/ExprMutator, StmtVisitor/StmtMutator,
 *        StmtExprVisitor/StmtExprMutator.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/ir/with_context.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <memory>
#include <string>

namespace tvm {
namespace tirx {

namespace {

ffi::Function MakeAnalyzerModule(std::shared_ptr<arith::Analyzer> analyzer) {
  using ffi::Function;
  using ffi::TypedFunction;
  auto f = [analyzer](std::string name) -> ffi::Function {
    if (name == "const_int_bound") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->const_int_bound(args[0].cast<PrimExpr>());
      });
    } else if (name == "modular_set") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->modular_set(args[0].cast<PrimExpr>());
      });
    } else if (name == "const_int_bound_update") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        analyzer->const_int_bound.Update(args[0].cast<Var>(), args[1].cast<arith::ConstIntBound>(),
                                         args[2].cast<bool>());
      });
    } else if (name == "const_int_bound_is_bound") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->const_int_bound.IsBound(args[0].cast<Var>());
      });
    } else if (name == "Simplify") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        if (args.size() == 1) {
          *ret = analyzer->Simplify(args[0].cast<PrimExpr>());
        } else if (args.size() == 2) {
          *ret = analyzer->Simplify(args[0].cast<PrimExpr>(), args[1].cast<int>());
        } else {
          TVM_FFI_THROW(InternalError) << "Invalid size of argument (" << args.size() << ")";
        }
      });
    } else if (name == "rewrite_simplify") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->rewrite_simplify(args[0].cast<PrimExpr>());
      });
    } else if (name == "get_rewrite_simplify_stats") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->rewrite_simplify.GetStatsCounters();
      });
    } else if (name == "reset_rewrite_simplify_stats") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        analyzer->rewrite_simplify.ResetStatsCounters();
      });
    } else if (name == "canonical_simplify") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->canonical_simplify(args[0].cast<PrimExpr>());
      });
    } else if (name == "int_set") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->int_set(args[0].cast<PrimExpr>(),
                                 args[1].cast<ffi::Map<Var, arith::IntSet>>());
      });
    } else if (name == "bind") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        bool allow_override = args.size() >= 3 && args[2].cast<bool>();
        if (auto opt_range = args[1].try_cast<Range>()) {
          analyzer->Bind(args[0].cast<Var>(), opt_range.value(), allow_override);
        } else {
          analyzer->Bind(args[0].cast<Var>(), args[1].cast<PrimExpr>(), allow_override);
        }
      });
    } else if (name == "can_prove") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        int strength = args[1].cast<int>();
        *ret = analyzer->CanProve(args[0].cast<PrimExpr>(),
                                  static_cast<arith::ProofStrength>(strength));
      });
    } else if (name == "enter_constraint_context") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        auto ctx = std::shared_ptr<With<arith::ConstraintContext>>(
            new With<arith::ConstraintContext>(analyzer.get(), args[0].cast<PrimExpr>()));
        auto fexit = [ctx](ffi::PackedArgs, ffi::Any*) mutable { ctx.reset(); };
        *ret = ffi::Function::FromPacked(fexit);
      });
    } else if (name == "can_prove_equal") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = analyzer->CanProveEqual(args[0].cast<PrimExpr>(), args[1].cast<PrimExpr>());
      });
    } else if (name == "get_enabled_extensions") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        *ret = static_cast<std::int64_t>(analyzer->rewrite_simplify.GetEnabledExtensions());
      });
    } else if (name == "set_enabled_extensions") {
      return ffi::Function([analyzer](ffi::PackedArgs args, ffi::Any* ret) {
        int64_t flags = args[0].cast<int64_t>();
        analyzer->rewrite_simplify.SetEnabledExtensions(
            static_cast<arith::RewriteSimplifier::Extension>(flags));
      });
    }
    return ffi::Function();
  };
  return ffi::TypedFunction<ffi::Function(std::string)>(f);
}

PrimExpr ExtractRealCondition(PrimExpr condition) {
  if (auto call = condition.as<CallNode>()) {
    if (call->op.same_as(builtin::likely())) {
      return call->args[0];
    }
  }
  return condition;
}

}  // namespace

// ================================================
// Helper Macros
// ================================================
#define PY_EXPR_VISITOR_DISPATCH(OP, PY_FUNC) \
  void VisitExpr_(const OP* op) override {    \
    if (PY_FUNC != nullptr) {                 \
      PY_FUNC(op);                            \
    } else {                                  \
      StmtExprVisitor::VisitExpr_(op);        \
    }                                         \
  }

#define IR_EXPR_VISITOR_DEFAULT_DISPATCH(OP)                                  \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitExpr_(static_cast<const OP*>(n.get()));       \
  });

#define PY_STMT_VISITOR_DISPATCH(OP, PY_FUNC) \
  void VisitStmt_(const OP* op) override {    \
    if (PY_FUNC != nullptr) {                 \
      PY_FUNC(op);                            \
    } else {                                  \
      StmtExprVisitor::VisitStmt_(op);        \
    }                                         \
  }

#define PY_STMT_VISITOR_DEFAULT_DISPATCH(OP)                                  \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitStmt_(static_cast<const OP*>(n.get()));       \
  });

#define PY_EXPR_MUTATOR_DISPATCH(OP, PY_FUNC)  \
  PrimExpr VisitExpr_(const OP* op) override { \
    if (PY_FUNC != nullptr) {                  \
      return PY_FUNC(op).cast<PrimExpr>();     \
    } else {                                   \
      return StmtExprMutator::VisitExpr_(op);  \
    }                                          \
  }

#define PY_EXPR_MUTATOR_DEFAULT_DISPATCH(OP)                                   \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) {  \
    return self->StmtExprMutator::VisitExpr_(static_cast<const OP*>(n.get())); \
  });

#define PY_STMT_MUTATOR_DISPATCH(OP, PY_FUNC) \
  Stmt VisitStmt_(const OP* op) override {    \
    if (PY_FUNC != nullptr) {                 \
      return PY_FUNC(op).cast<Stmt>();        \
    } else {                                  \
      return StmtExprMutator::VisitStmt_(op); \
    }                                         \
  }

#define PY_STMT_MUTATOR_DEFAULT_DISPATCH(OP)                                   \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) {  \
    return self->StmtExprMutator::VisitStmt_(static_cast<const OP*>(n.get())); \
  });

/*! \brief The python interface of StmtExprVisitor. */
class PyStmtExprVisitorNode : public ffi::Object, public StmtExprVisitor {
 private:
  using TSelf = PyStmtExprVisitorNode;
  using FExprType = tvm::NodeFunctor<void(const ffi::ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<void(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  // Expression functions
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  ffi::Function f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarNode* op)` function. */
  ffi::Function f_visit_var{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SizeVarNode* op)` function. */
  ffi::Function f_visit_size_var{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const BufferLoadNode* op)` function. */
  ffi::Function f_visit_buffer_load{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ProducerLoadNode* op)` function. */
  ffi::Function f_visit_producer_load{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LetNode* op)` function. */
  ffi::Function f_visit_let{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallNode* op)` function. */
  ffi::Function f_visit_call{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const AddNode* op)` function. */
  ffi::Function f_visit_add{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SubNode* op)` function. */
  ffi::Function f_visit_sub{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MulNode* op)` function. */
  ffi::Function f_visit_mul{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DivNode* op)` function. */
  ffi::Function f_visit_div{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ModNode* op)` function. */
  ffi::Function f_visit_mod{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloorDivNode* op)` function. */
  ffi::Function f_visit_floor_div{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloorModNode* op)` function. */
  ffi::Function f_visit_floor_mod{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MinNode* op)` function. */
  ffi::Function f_visit_min{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MaxNode* op)` function. */
  ffi::Function f_visit_max{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const EQNode* op)` function. */
  ffi::Function f_visit_eq{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const NENode* op)` function. */
  ffi::Function f_visit_ne{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LTNode* op)` function. */
  ffi::Function f_visit_lt{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LENode* op)` function. */
  ffi::Function f_visit_le{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GTNode* op)` function. */
  ffi::Function f_visit_gt{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GENode* op)` function. */
  ffi::Function f_visit_ge{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const AndNode* op)` function. */
  ffi::Function f_visit_and{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OrNode* op)` function. */
  ffi::Function f_visit_or{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ReduceNode* op)` function. */
  ffi::Function f_visit_reduce{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CastNode* op)` function. */
  ffi::Function f_visit_cast{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const NotNode* op)` function. */
  ffi::Function f_visit_not{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SelectNode* op)` function. */
  ffi::Function f_visit_select{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const RampNode* op)` function. */
  ffi::Function f_visit_ramp{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const BroadcastNode* op)` function. */
  ffi::Function f_visit_broadcast{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShuffleNode* op)` function. */
  ffi::Function f_visit_shuffle{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IntImmNode* op)` function. */
  ffi::Function f_visit_int_imm{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloatImmNode* op)` function. */
  ffi::Function f_visit_float_imm{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmNode* op)` function. */
  ffi::Function f_visit_string_imm{nullptr};

  // Statement functions
  /*! \brief The packed function to the `VisitStmt(const Stmt& stmt)` function. */
  ffi::Function f_visit_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BindNode* op)` function. */
  ffi::Function f_visit_bind{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AttrStmtNode* op)` function. */
  ffi::Function f_visit_attr_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const IfThenElseNode* op)` function. */
  ffi::Function f_visit_if_then_else{nullptr};  // NOLINT(readability/braces)
  /*! \brief The packed function to the `VisitStmt_(const ForNode* op)` function. */
  ffi::Function f_visit_for{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const WhileNode* op)` function. */
  ffi::Function f_visit_while{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocBufferNode* op)` function. */
  ffi::Function f_visit_alloc_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const DeclBufferNode* op)` function. */
  ffi::Function f_visit_decl_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferStoreNode* op)` function. */
  ffi::Function f_visit_buffer_store{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AssertStmtNode* op)` function. */
  ffi::Function f_visit_assert_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SeqStmtNode* op)` function. */
  ffi::Function f_visit_seq_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const EvaluateNode* op)` function. */
  ffi::Function f_visit_evaluate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SBlockNode* op)` function. */
  ffi::Function f_visit_block{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SBlockRealizeNode* op)` function. */
  ffi::Function f_visit_sblock_realize{nullptr};

  using StmtExprVisitor::VisitExpr;
  using StmtExprVisitor::VisitStmt;

  void DefaultVisitExpr(const PrimExpr& expr) {
    static FExprType vtable = InitExprVTable();
    vtable(expr, this);
  }

  void DefaultVisitStmt(const Stmt& stmt) {
    static FStmtType vtable = InitStmtVTable();
    vtable(stmt, this);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyStmtExprVisitorNode>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.PyStmtExprVisitor", PyStmtExprVisitorNode, ffi::Object);

 private:
  // Statement functions
  PY_STMT_VISITOR_DISPATCH(BindNode, f_visit_bind);
  PY_STMT_VISITOR_DISPATCH(AttrStmtNode, f_visit_attr_stmt);
  PY_STMT_VISITOR_DISPATCH(IfThenElseNode, f_visit_if_then_else);
  PY_STMT_VISITOR_DISPATCH(ForNode, f_visit_for);
  PY_STMT_VISITOR_DISPATCH(WhileNode, f_visit_while);
  PY_STMT_VISITOR_DISPATCH(AllocBufferNode, f_visit_alloc_buffer);
  PY_STMT_VISITOR_DISPATCH(DeclBufferNode, f_visit_decl_buffer);
  PY_STMT_VISITOR_DISPATCH(BufferStoreNode, f_visit_buffer_store);
  PY_STMT_VISITOR_DISPATCH(AssertStmtNode, f_visit_assert_stmt);
  PY_STMT_VISITOR_DISPATCH(SeqStmtNode, f_visit_seq_stmt);
  PY_STMT_VISITOR_DISPATCH(EvaluateNode, f_visit_evaluate);
  PY_STMT_VISITOR_DISPATCH(SBlockNode, f_visit_block);
  PY_STMT_VISITOR_DISPATCH(SBlockRealizeNode, f_visit_sblock_realize);
  // Expression functions
  PY_EXPR_VISITOR_DISPATCH(VarNode, f_visit_var);
  PY_EXPR_VISITOR_DISPATCH(SizeVarNode, f_visit_size_var);
  PY_EXPR_VISITOR_DISPATCH(BufferLoadNode, f_visit_buffer_load);
  PY_EXPR_VISITOR_DISPATCH(ProducerLoadNode, f_visit_producer_load);
  PY_EXPR_VISITOR_DISPATCH(LetNode, f_visit_let);
  PY_EXPR_VISITOR_DISPATCH(CallNode, f_visit_call);
  PY_EXPR_VISITOR_DISPATCH(AddNode, f_visit_add);
  PY_EXPR_VISITOR_DISPATCH(SubNode, f_visit_sub);
  PY_EXPR_VISITOR_DISPATCH(MulNode, f_visit_mul);
  PY_EXPR_VISITOR_DISPATCH(DivNode, f_visit_div);
  PY_EXPR_VISITOR_DISPATCH(ModNode, f_visit_mod);
  PY_EXPR_VISITOR_DISPATCH(FloorDivNode, f_visit_floor_div);
  PY_EXPR_VISITOR_DISPATCH(FloorModNode, f_visit_floor_mod);
  PY_EXPR_VISITOR_DISPATCH(MinNode, f_visit_min);
  PY_EXPR_VISITOR_DISPATCH(MaxNode, f_visit_max);
  PY_EXPR_VISITOR_DISPATCH(EQNode, f_visit_eq);
  PY_EXPR_VISITOR_DISPATCH(NENode, f_visit_ne);
  PY_EXPR_VISITOR_DISPATCH(LTNode, f_visit_lt);
  PY_EXPR_VISITOR_DISPATCH(LENode, f_visit_le);
  PY_EXPR_VISITOR_DISPATCH(GTNode, f_visit_gt);
  PY_EXPR_VISITOR_DISPATCH(GENode, f_visit_ge);
  PY_EXPR_VISITOR_DISPATCH(AndNode, f_visit_and);
  PY_EXPR_VISITOR_DISPATCH(OrNode, f_visit_or);
  PY_EXPR_VISITOR_DISPATCH(ReduceNode, f_visit_reduce);
  PY_EXPR_VISITOR_DISPATCH(CastNode, f_visit_cast);
  PY_EXPR_VISITOR_DISPATCH(NotNode, f_visit_not);
  PY_EXPR_VISITOR_DISPATCH(SelectNode, f_visit_select);
  PY_EXPR_VISITOR_DISPATCH(RampNode, f_visit_ramp);
  PY_EXPR_VISITOR_DISPATCH(BroadcastNode, f_visit_broadcast);
  PY_EXPR_VISITOR_DISPATCH(ShuffleNode, f_visit_shuffle);
  PY_EXPR_VISITOR_DISPATCH(IntImmNode, f_visit_int_imm);
  PY_EXPR_VISITOR_DISPATCH(FloatImmNode, f_visit_float_imm);
  PY_EXPR_VISITOR_DISPATCH(StringImmNode, f_visit_string_imm);

 private:
  static FExprType InitExprVTable() {
    FExprType vtable;
    // Set dispatch
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(VarNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(SizeVarNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(BufferLoadNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(ProducerLoadNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(LetNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(CallNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(AddNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(SubNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(MulNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(DivNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(ModNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(FloorDivNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(FloorModNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(MinNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(MaxNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(EQNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(NENode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(LTNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(LENode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(GTNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(GENode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(AndNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(OrNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(ReduceNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(CastNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(NotNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(SelectNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(RampNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(ShuffleNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(BroadcastNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(IntImmNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(FloatImmNode);
    IR_EXPR_VISITOR_DEFAULT_DISPATCH(StringImmNode);
    vtable.Finalize();
    return vtable;
  }

  static FStmtType InitStmtVTable() {
    FStmtType vtable;
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BindNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AttrStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(IfThenElseNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(ForNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(WhileNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AllocBufferNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(DeclBufferNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BufferStoreNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AssertStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(SeqStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(EvaluateNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(SBlockNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(SBlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief Managed reference to PyStmtExprVisitorNode.
 * \sa PyStmtExprVisitorNode
 */
class PyStmtExprVisitor : public ffi::ObjectRef {
 public:
  explicit PyStmtExprVisitor(ffi::ObjectPtr<PyStmtExprVisitorNode> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_DLL static PyStmtExprVisitor MakePyStmtExprVisitor(ffi::Function f_visit_stmt,            //
                                                         ffi::Function f_visit_expr,            //
                                                         ffi::Function f_visit_bind,            //
                                                         ffi::Function f_visit_attr_stmt,       //
                                                         ffi::Function f_visit_if_then_else,    //
                                                         ffi::Function f_visit_for,             //
                                                         ffi::Function f_visit_while,           //
                                                         ffi::Function f_visit_alloc_buffer,    //
                                                         ffi::Function f_visit_decl_buffer,     //
                                                         ffi::Function f_visit_buffer_store,    //
                                                         ffi::Function f_visit_assert_stmt,     //
                                                         ffi::Function f_visit_seq_stmt,        //
                                                         ffi::Function f_visit_evaluate,        //
                                                         ffi::Function f_visit_block,           //
                                                         ffi::Function f_visit_sblock_realize,  //
                                                         ffi::Function f_visit_var,             //
                                                         ffi::Function f_visit_size_var,        //
                                                         ffi::Function f_visit_buffer_load,     //
                                                         ffi::Function f_visit_producer_load,   //
                                                         ffi::Function f_visit_let,             //
                                                         ffi::Function f_visit_call,            //
                                                         ffi::Function f_visit_add,             //
                                                         ffi::Function f_visit_sub,             //
                                                         ffi::Function f_visit_mul,             //
                                                         ffi::Function f_visit_div,             //
                                                         ffi::Function f_visit_mod,             //
                                                         ffi::Function f_visit_floor_div,       //
                                                         ffi::Function f_visit_floor_mod,       //
                                                         ffi::Function f_visit_min,             //
                                                         ffi::Function f_visit_max,             //
                                                         ffi::Function f_visit_eq,              //
                                                         ffi::Function f_visit_ne,              //
                                                         ffi::Function f_visit_lt,              //
                                                         ffi::Function f_visit_le,              //
                                                         ffi::Function f_visit_gt,              //
                                                         ffi::Function f_visit_ge,              //
                                                         ffi::Function f_visit_and,             //
                                                         ffi::Function f_visit_or,              //
                                                         ffi::Function f_visit_reduce,          //
                                                         ffi::Function f_visit_cast,            //
                                                         ffi::Function f_visit_not,             //
                                                         ffi::Function f_visit_select,          //
                                                         ffi::Function f_visit_ramp,            //
                                                         ffi::Function f_visit_broadcast,       //
                                                         ffi::Function f_visit_shuffle,         //
                                                         ffi::Function f_visit_int_imm,         //
                                                         ffi::Function f_visit_float_imm,       //
                                                         ffi::Function f_visit_string_imm) {
    ffi::ObjectPtr<PyStmtExprVisitorNode> n = ffi::make_object<PyStmtExprVisitorNode>();
    n->f_visit_stmt = std::move(f_visit_stmt);
    n->f_visit_expr = std::move(f_visit_expr);
    // Set statement functions
    n->f_visit_bind = std::move(f_visit_bind);
    n->f_visit_attr_stmt = std::move(f_visit_attr_stmt);
    n->f_visit_if_then_else = std::move(f_visit_if_then_else);
    n->f_visit_for = std::move(f_visit_for);
    n->f_visit_while = std::move(f_visit_while);
    n->f_visit_alloc_buffer = std::move(f_visit_alloc_buffer);
    n->f_visit_decl_buffer = std::move(f_visit_decl_buffer);
    n->f_visit_buffer_store = std::move(f_visit_buffer_store);
    n->f_visit_assert_stmt = std::move(f_visit_assert_stmt);
    n->f_visit_seq_stmt = std::move(f_visit_seq_stmt);
    n->f_visit_evaluate = std::move(f_visit_evaluate);
    n->f_visit_block = std::move(f_visit_block);
    n->f_visit_sblock_realize = std::move(f_visit_sblock_realize);
    // Set expression functions
    n->f_visit_var = std::move(f_visit_var);
    n->f_visit_size_var = std::move(f_visit_size_var);
    n->f_visit_buffer_load = std::move(f_visit_buffer_load);
    n->f_visit_producer_load = std::move(f_visit_producer_load);
    n->f_visit_let = std::move(f_visit_let);
    n->f_visit_call = std::move(f_visit_call);
    n->f_visit_add = std::move(f_visit_add);
    n->f_visit_sub = std::move(f_visit_sub);
    n->f_visit_mul = std::move(f_visit_mul);
    n->f_visit_div = std::move(f_visit_div);
    n->f_visit_mod = std::move(f_visit_mod);
    n->f_visit_floor_div = std::move(f_visit_floor_div);
    n->f_visit_floor_mod = std::move(f_visit_floor_mod);
    n->f_visit_min = std::move(f_visit_min);
    n->f_visit_max = std::move(f_visit_max);
    n->f_visit_eq = std::move(f_visit_eq);
    n->f_visit_ne = std::move(f_visit_ne);
    n->f_visit_lt = std::move(f_visit_lt);
    n->f_visit_le = std::move(f_visit_le);
    n->f_visit_gt = std::move(f_visit_gt);
    n->f_visit_ge = std::move(f_visit_ge);
    n->f_visit_and = std::move(f_visit_and);
    n->f_visit_or = std::move(f_visit_or);
    n->f_visit_reduce = std::move(f_visit_reduce);
    n->f_visit_cast = std::move(f_visit_cast);
    n->f_visit_not = std::move(f_visit_not);
    n->f_visit_select = std::move(f_visit_select);
    n->f_visit_ramp = std::move(f_visit_ramp);
    n->f_visit_broadcast = std::move(f_visit_broadcast);
    n->f_visit_shuffle = std::move(f_visit_shuffle);
    n->f_visit_int_imm = std::move(f_visit_int_imm);
    n->f_visit_float_imm = std::move(f_visit_float_imm);
    n->f_visit_string_imm = std::move(f_visit_string_imm);
    return PyStmtExprVisitor(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyStmtExprVisitor, ffi::ObjectRef,
                                                PyStmtExprVisitorNode);
};

/*! \brief The python interface of StmtExprMutator. */
class PyStmtExprMutatorNode : public ffi::Object, public StmtExprMutator {
 private:
  using TSelf = PyStmtExprMutatorNode;
  using FExprType = tvm::NodeFunctor<PrimExpr(const ffi::ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<Stmt(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  // Expression functions
  /*! \brief The packed function to the `VisitExpr(const Expr& expr)` function. */
  ffi::Function f_visit_expr{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const VarNode* op)` function. */
  ffi::Function f_visit_var{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SizeVarNode* op)` function. */
  ffi::Function f_visit_size_var{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const BufferLoadNode* op)` function. */
  ffi::Function f_visit_buffer_load{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ProducerLoadNode* op)` function. */
  ffi::Function f_visit_producer_load{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LetNode* op)` function. */
  ffi::Function f_visit_let{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CallNode* op)` function. */
  ffi::Function f_visit_call{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const AddNode* op)` function. */
  ffi::Function f_visit_add{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SubNode* op)` function. */
  ffi::Function f_visit_sub{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MulNode* op)` function. */
  ffi::Function f_visit_mul{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const DivNode* op)` function. */
  ffi::Function f_visit_div{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ModNode* op)` function. */
  ffi::Function f_visit_mod{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloorDivNode* op)` function. */
  ffi::Function f_visit_floor_div{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloorModNode* op)` function. */
  ffi::Function f_visit_floor_mod{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MinNode* op)` function. */
  ffi::Function f_visit_min{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const MaxNode* op)` function. */
  ffi::Function f_visit_max{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const EQNode* op)` function. */
  ffi::Function f_visit_eq{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const NENode* op)` function. */
  ffi::Function f_visit_ne{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LTNode* op)` function. */
  ffi::Function f_visit_lt{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const LENode* op)` function. */
  ffi::Function f_visit_le{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GTNode* op)` function. */
  ffi::Function f_visit_gt{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const GENode* op)` function. */
  ffi::Function f_visit_ge{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const AndNode* op)` function. */
  ffi::Function f_visit_and{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const OrNode* op)` function. */
  ffi::Function f_visit_or{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ReduceNode* op)` function. */
  ffi::Function f_visit_reduce{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const CastNode* op)` function. */
  ffi::Function f_visit_cast{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const NotNode* op)` function. */
  ffi::Function f_visit_not{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const SelectNode* op)` function. */
  ffi::Function f_visit_select{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const RampNode* op)` function. */
  ffi::Function f_visit_ramp{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const BroadcastNode* op)` function. */
  ffi::Function f_visit_broadcast{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const ShuffleNode* op)` function. */
  ffi::Function f_visit_shuffle{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const IntImmNode* op)` function. */
  ffi::Function f_visit_int_imm{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const FloatImmNode* op)` function. */
  ffi::Function f_visit_float_imm{nullptr};
  /*! \brief The packed function to the `VisitExpr_(const StringImmNode* op)` function. */
  ffi::Function f_visit_string_imm{nullptr};

  // Statement functions
  /*! \brief The packed function to the `VisitStmt(const Stmt& stmt)` function. */
  ffi::Function f_visit_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BindNode* op)` function. */
  ffi::Function f_visit_bind{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AttrStmtNode* op)` function. */
  ffi::Function f_visit_attr_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const IfThenElseNode* op)` function. */
  ffi::Function f_visit_if_then_else{nullptr};  // NOLINT(readability/braces)
  /*! \brief The packed function to the `VisitStmt_(const ForNode* op)` function. */
  ffi::Function f_visit_for{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const WhileNode* op)` function. */
  ffi::Function f_visit_while{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocBufferNode* op)` function. */
  ffi::Function f_visit_alloc_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const DeclBufferNode* op)` function. */
  ffi::Function f_visit_decl_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferStoreNode* op)` function. */
  ffi::Function f_visit_buffer_store{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AssertStmtNode* op)` function. */
  ffi::Function f_visit_assert_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SeqStmtNode* op)` function. */
  ffi::Function f_visit_seq_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const EvaluateNode* op)` function. */
  ffi::Function f_visit_evaluate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SBlockNode* op)` function. */
  ffi::Function f_visit_block{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SBlockRealizeNode* op)` function. */
  ffi::Function f_visit_sblock_realize{nullptr};

  using StmtExprMutator::VisitExpr;
  using StmtExprMutator::VisitStmt;

  void DefaultVisitExpr(const PrimExpr& expr) {
    static FExprType vtable = InitExprVTable();
    vtable(expr, this);
  }

  void DefaultVisitStmt(const Stmt& stmt) {
    static FStmtType vtable = InitStmtVTable();
    vtable(stmt, this);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyStmtExprMutatorNode>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.PyStmtExprMutator", PyStmtExprMutatorNode, ffi::Object);

 private:
  // Statement functions
  PY_STMT_MUTATOR_DISPATCH(BindNode, f_visit_bind);
  PY_STMT_MUTATOR_DISPATCH(AttrStmtNode, f_visit_attr_stmt);
  PY_STMT_MUTATOR_DISPATCH(IfThenElseNode, f_visit_if_then_else);
  PY_STMT_MUTATOR_DISPATCH(ForNode, f_visit_for);
  PY_STMT_MUTATOR_DISPATCH(WhileNode, f_visit_while);
  PY_STMT_MUTATOR_DISPATCH(AllocBufferNode, f_visit_alloc_buffer);
  PY_STMT_MUTATOR_DISPATCH(DeclBufferNode, f_visit_decl_buffer);
  PY_STMT_MUTATOR_DISPATCH(BufferStoreNode, f_visit_buffer_store);
  PY_STMT_MUTATOR_DISPATCH(AssertStmtNode, f_visit_assert_stmt);
  PY_STMT_MUTATOR_DISPATCH(SeqStmtNode, f_visit_seq_stmt);
  PY_STMT_MUTATOR_DISPATCH(EvaluateNode, f_visit_evaluate);
  PY_STMT_MUTATOR_DISPATCH(SBlockNode, f_visit_block);
  PY_STMT_MUTATOR_DISPATCH(SBlockRealizeNode, f_visit_sblock_realize);
  // Expression functions
  PY_EXPR_MUTATOR_DISPATCH(VarNode, f_visit_var);
  PY_EXPR_MUTATOR_DISPATCH(SizeVarNode, f_visit_size_var);
  PY_EXPR_MUTATOR_DISPATCH(BufferLoadNode, f_visit_buffer_load);
  PY_EXPR_MUTATOR_DISPATCH(ProducerLoadNode, f_visit_producer_load);
  PY_EXPR_MUTATOR_DISPATCH(LetNode, f_visit_let);
  PY_EXPR_MUTATOR_DISPATCH(CallNode, f_visit_call);
  PY_EXPR_MUTATOR_DISPATCH(AddNode, f_visit_add);
  PY_EXPR_MUTATOR_DISPATCH(SubNode, f_visit_sub);
  PY_EXPR_MUTATOR_DISPATCH(MulNode, f_visit_mul);
  PY_EXPR_MUTATOR_DISPATCH(DivNode, f_visit_div);
  PY_EXPR_MUTATOR_DISPATCH(ModNode, f_visit_mod);
  PY_EXPR_MUTATOR_DISPATCH(FloorDivNode, f_visit_floor_div);
  PY_EXPR_MUTATOR_DISPATCH(FloorModNode, f_visit_floor_mod);
  PY_EXPR_MUTATOR_DISPATCH(MinNode, f_visit_min);
  PY_EXPR_MUTATOR_DISPATCH(MaxNode, f_visit_max);
  PY_EXPR_MUTATOR_DISPATCH(EQNode, f_visit_eq);
  PY_EXPR_MUTATOR_DISPATCH(NENode, f_visit_ne);
  PY_EXPR_MUTATOR_DISPATCH(LTNode, f_visit_lt);
  PY_EXPR_MUTATOR_DISPATCH(LENode, f_visit_le);
  PY_EXPR_MUTATOR_DISPATCH(GTNode, f_visit_gt);
  PY_EXPR_MUTATOR_DISPATCH(GENode, f_visit_ge);
  PY_EXPR_MUTATOR_DISPATCH(AndNode, f_visit_and);
  PY_EXPR_MUTATOR_DISPATCH(OrNode, f_visit_or);
  PY_EXPR_MUTATOR_DISPATCH(ReduceNode, f_visit_reduce);
  PY_EXPR_MUTATOR_DISPATCH(CastNode, f_visit_cast);
  PY_EXPR_MUTATOR_DISPATCH(NotNode, f_visit_not);
  PY_EXPR_MUTATOR_DISPATCH(SelectNode, f_visit_select);
  PY_EXPR_MUTATOR_DISPATCH(RampNode, f_visit_ramp);
  PY_EXPR_MUTATOR_DISPATCH(BroadcastNode, f_visit_broadcast);
  PY_EXPR_MUTATOR_DISPATCH(ShuffleNode, f_visit_shuffle);
  PY_EXPR_MUTATOR_DISPATCH(IntImmNode, f_visit_int_imm);
  PY_EXPR_MUTATOR_DISPATCH(FloatImmNode, f_visit_float_imm);
  PY_EXPR_MUTATOR_DISPATCH(StringImmNode, f_visit_string_imm);

 private:
  static FExprType InitExprVTable() {
    FExprType vtable;
    // Set dispatch
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(VarNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(SizeVarNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(BufferLoadNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(ProducerLoadNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(LetNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(CallNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(AddNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(SubNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(MulNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(DivNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(ModNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(FloorDivNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(FloorModNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(MinNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(MaxNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(EQNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(NENode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(LTNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(LENode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(GTNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(GENode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(AndNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(OrNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(ReduceNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(CastNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(NotNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(SelectNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(RampNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(ShuffleNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(BroadcastNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(IntImmNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(FloatImmNode);
    PY_EXPR_MUTATOR_DEFAULT_DISPATCH(StringImmNode);
    vtable.Finalize();
    return vtable;
  }

  static FStmtType InitStmtVTable() {
    FStmtType vtable;
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BindNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AttrStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(IfThenElseNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(ForNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(WhileNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AllocBufferNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(DeclBufferNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BufferStoreNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AssertStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(SeqStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(EvaluateNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(SBlockNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(SBlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }
};

/*! \brief Managed reference to PyStmtExprMutatorNode. */
class PyStmtExprMutator : public ffi::ObjectRef {
 public:
  explicit PyStmtExprMutator(ffi::ObjectPtr<PyStmtExprMutatorNode> data) : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  /*!
   * \brief Create a PyStmtExprMutator with customized methods on the python-side.
   * \return The PyStmtExprMutator created.
   */
  TVM_DLL static PyStmtExprMutator MakePyStmtExprMutator(ffi::Function f_visit_stmt,            //
                                                         ffi::Function f_visit_expr,            //
                                                         ffi::Function f_visit_bind,            //
                                                         ffi::Function f_visit_attr_stmt,       //
                                                         ffi::Function f_visit_if_then_else,    //
                                                         ffi::Function f_visit_for,             //
                                                         ffi::Function f_visit_while,           //
                                                         ffi::Function f_visit_alloc_buffer,    //
                                                         ffi::Function f_visit_decl_buffer,     //
                                                         ffi::Function f_visit_buffer_store,    //
                                                         ffi::Function f_visit_assert_stmt,     //
                                                         ffi::Function f_visit_seq_stmt,        //
                                                         ffi::Function f_visit_evaluate,        //
                                                         ffi::Function f_visit_block,           //
                                                         ffi::Function f_visit_sblock_realize,  //
                                                         ffi::Function f_visit_var,             //
                                                         ffi::Function f_visit_size_var,        //
                                                         ffi::Function f_visit_buffer_load,     //
                                                         ffi::Function f_visit_producer_load,   //
                                                         ffi::Function f_visit_let,             //
                                                         ffi::Function f_visit_call,            //
                                                         ffi::Function f_visit_add,             //
                                                         ffi::Function f_visit_sub,             //
                                                         ffi::Function f_visit_mul,             //
                                                         ffi::Function f_visit_div,             //
                                                         ffi::Function f_visit_mod,             //
                                                         ffi::Function f_visit_floor_div,       //
                                                         ffi::Function f_visit_floor_mod,       //
                                                         ffi::Function f_visit_min,             //
                                                         ffi::Function f_visit_max,             //
                                                         ffi::Function f_visit_eq,              //
                                                         ffi::Function f_visit_ne,              //
                                                         ffi::Function f_visit_lt,              //
                                                         ffi::Function f_visit_le,              //
                                                         ffi::Function f_visit_gt,              //
                                                         ffi::Function f_visit_ge,              //
                                                         ffi::Function f_visit_and,             //
                                                         ffi::Function f_visit_or,              //
                                                         ffi::Function f_visit_reduce,          //
                                                         ffi::Function f_visit_cast,            //
                                                         ffi::Function f_visit_not,             //
                                                         ffi::Function f_visit_select,          //
                                                         ffi::Function f_visit_ramp,            //
                                                         ffi::Function f_visit_broadcast,       //
                                                         ffi::Function f_visit_shuffle,         //
                                                         ffi::Function f_visit_int_imm,         //
                                                         ffi::Function f_visit_float_imm,       //
                                                         ffi::Function f_visit_string_imm) {
    ffi::ObjectPtr<PyStmtExprMutatorNode> n = ffi::make_object<PyStmtExprMutatorNode>();
    n->f_visit_stmt = std::move(f_visit_stmt);
    n->f_visit_expr = std::move(f_visit_expr);
    // Statement functions
    n->f_visit_bind = std::move(f_visit_bind);
    n->f_visit_attr_stmt = std::move(f_visit_attr_stmt);
    n->f_visit_if_then_else = std::move(f_visit_if_then_else);
    n->f_visit_for = std::move(f_visit_for);
    n->f_visit_while = std::move(f_visit_while);
    n->f_visit_alloc_buffer = std::move(f_visit_alloc_buffer);
    n->f_visit_decl_buffer = std::move(f_visit_decl_buffer);
    n->f_visit_buffer_store = std::move(f_visit_buffer_store);
    n->f_visit_assert_stmt = std::move(f_visit_assert_stmt);
    n->f_visit_seq_stmt = std::move(f_visit_seq_stmt);
    n->f_visit_evaluate = std::move(f_visit_evaluate);
    n->f_visit_block = std::move(f_visit_block);
    n->f_visit_sblock_realize = std::move(f_visit_sblock_realize);
    // Expression functions
    n->f_visit_var = std::move(f_visit_var);
    n->f_visit_size_var = std::move(f_visit_size_var);
    n->f_visit_buffer_load = std::move(f_visit_buffer_load);
    n->f_visit_producer_load = std::move(f_visit_producer_load);
    n->f_visit_let = std::move(f_visit_let);
    n->f_visit_call = std::move(f_visit_call);
    n->f_visit_add = std::move(f_visit_add);
    n->f_visit_sub = std::move(f_visit_sub);
    n->f_visit_mul = std::move(f_visit_mul);
    n->f_visit_div = std::move(f_visit_div);
    n->f_visit_mod = std::move(f_visit_mod);
    n->f_visit_floor_div = std::move(f_visit_floor_div);
    n->f_visit_floor_mod = std::move(f_visit_floor_mod);
    n->f_visit_min = std::move(f_visit_min);
    n->f_visit_max = std::move(f_visit_max);
    n->f_visit_eq = std::move(f_visit_eq);
    n->f_visit_ne = std::move(f_visit_ne);
    n->f_visit_lt = std::move(f_visit_lt);
    n->f_visit_le = std::move(f_visit_le);
    n->f_visit_gt = std::move(f_visit_gt);
    n->f_visit_ge = std::move(f_visit_ge);
    n->f_visit_and = std::move(f_visit_and);
    n->f_visit_or = std::move(f_visit_or);
    n->f_visit_reduce = std::move(f_visit_reduce);
    n->f_visit_cast = std::move(f_visit_cast);
    n->f_visit_not = std::move(f_visit_not);
    n->f_visit_select = std::move(f_visit_select);
    n->f_visit_ramp = std::move(f_visit_ramp);
    n->f_visit_broadcast = std::move(f_visit_broadcast);
    n->f_visit_shuffle = std::move(f_visit_shuffle);
    n->f_visit_int_imm = std::move(f_visit_int_imm);
    n->f_visit_float_imm = std::move(f_visit_float_imm);
    n->f_visit_string_imm = std::move(f_visit_string_imm);
    return PyStmtExprMutator(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyStmtExprMutator, ffi::ObjectRef,
                                                PyStmtExprMutatorNode);
};

#define PY_STMT_EXPR_FUNCTOR_CALLBACKS(V) \
  V(f_visit_stmt)                         \
  V(f_visit_expr)                         \
  V(f_visit_bind)                         \
  V(f_visit_attr_stmt)                    \
  V(f_visit_if_then_else)                 \
  V(f_visit_for)                          \
  V(f_visit_while)                        \
  V(f_visit_alloc_buffer)                 \
  V(f_visit_decl_buffer)                  \
  V(f_visit_buffer_store)                 \
  V(f_visit_assert_stmt)                  \
  V(f_visit_seq_stmt)                     \
  V(f_visit_evaluate)                     \
  V(f_visit_block)                        \
  V(f_visit_sblock_realize)               \
  V(f_visit_var)                          \
  V(f_visit_size_var)                     \
  V(f_visit_buffer_load)                  \
  V(f_visit_producer_load)                \
  V(f_visit_let)                          \
  V(f_visit_call)                         \
  V(f_visit_add)                          \
  V(f_visit_sub)                          \
  V(f_visit_mul)                          \
  V(f_visit_div)                          \
  V(f_visit_mod)                          \
  V(f_visit_floor_div)                    \
  V(f_visit_floor_mod)                    \
  V(f_visit_min)                          \
  V(f_visit_max)                          \
  V(f_visit_eq)                           \
  V(f_visit_ne)                           \
  V(f_visit_lt)                           \
  V(f_visit_le)                           \
  V(f_visit_gt)                           \
  V(f_visit_ge)                           \
  V(f_visit_and)                          \
  V(f_visit_or)                           \
  V(f_visit_reduce)                       \
  V(f_visit_cast)                         \
  V(f_visit_not)                          \
  V(f_visit_select)                       \
  V(f_visit_ramp)                         \
  V(f_visit_broadcast)                    \
  V(f_visit_shuffle)                      \
  V(f_visit_int_imm)                      \
  V(f_visit_float_imm)                    \
  V(f_visit_string_imm)

template <typename TNode>
void SetStmtExprFunctorCallbacks(TNode* node,
                                 const ffi::Array<ffi::Optional<ffi::Function>>& callbacks) {
  int index = 0;
#define SET_CALLBACK(FIELD) node->FIELD = callbacks[index++].value_or(ffi::Function(nullptr));
  PY_STMT_EXPR_FUNCTOR_CALLBACKS(SET_CALLBACK)
#undef SET_CALLBACK
  TVM_FFI_ICHECK_EQ(index, callbacks.size());
}

#define PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(OP, METHOD)                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->METHOD(static_cast<const OP*>(n.get()));                            \
  });

#define PY_ANALYZER_EXPR_VISITOR_DEFAULT_DISPATCH(OP, METHOD)                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->METHOD(static_cast<const OP*>(n.get()));                            \
  });

#define PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(OP)                            \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitStmt_(static_cast<const OP*>(n.get()));       \
  });

#define PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(OP)                            \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitExpr_(static_cast<const OP*>(n.get()));       \
  });

#define PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(OP, METHOD)                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    return self->METHOD(static_cast<const OP*>(n.get()));                     \
  });

#define PY_ANALYZER_EXPR_MUTATOR_DEFAULT_DISPATCH(OP, METHOD)                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    return self->METHOD(static_cast<const OP*>(n.get()));                     \
  });

#define PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(OP)                             \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) {  \
    return self->StmtExprMutator::VisitStmt_(static_cast<const OP*>(n.get())); \
  });

#define PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(OP)                             \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) {  \
    return self->StmtExprMutator::VisitExpr_(static_cast<const OP*>(n.get())); \
  });

class PyStmtExprVisitorWithAnalyzerNode : public PyStmtExprVisitorNode {
 private:
  using TSelf = PyStmtExprVisitorWithAnalyzerNode;
  using FExprType = tvm::NodeFunctor<void(const ffi::ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<void(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  PyStmtExprVisitorWithAnalyzerNode() = default;
  PyStmtExprVisitorWithAnalyzerNode(const PyStmtExprVisitorWithAnalyzerNode& other)
      : PyStmtExprVisitorNode(other) {}

  ffi::Function GetAnalyzer() { return MakeAnalyzerModule(analyzer_); }

  void DefaultVisitExprWithAnalyzer(const PrimExpr& expr) {
    static FExprType vtable = InitExprVTable();
    vtable(expr, this);
  }

  void DefaultVisitStmtWithAnalyzer(const Stmt& stmt) {
    static FStmtType vtable = InitStmtVTable();
    vtable(stmt, this);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyStmtExprVisitorWithAnalyzerNode>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.PyStmtExprVisitorWithAnalyzer",
                              PyStmtExprVisitorWithAnalyzerNode, PyStmtExprVisitorNode);

 private:
  std::shared_ptr<arith::Analyzer> analyzer_{std::make_shared<arith::Analyzer>()};
  ScopeStack<WithGroup<arith::ConstraintContext>> constraint_scope_;

  void VisitStmt_(const ForNode* op) override {
    if (f_visit_for != nullptr) {
      f_visit_for(op);
    } else {
      DefaultVisitFor(op);
    }
  }

  void VisitStmt_(const SBlockNode* op) override {
    if (f_visit_block != nullptr) {
      f_visit_block(op);
    } else {
      DefaultVisitSBlock(op);
    }
  }

  void VisitStmt_(const BindNode* op) override {
    if (f_visit_bind != nullptr) {
      f_visit_bind(op);
    } else {
      DefaultVisitBind(op);
    }
  }

  void VisitStmt_(const IfThenElseNode* op) override {
    if (f_visit_if_then_else != nullptr) {
      f_visit_if_then_else(op);
    } else {
      DefaultVisitIfThenElse(op);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) override {
    if (f_visit_attr_stmt != nullptr) {
      f_visit_attr_stmt(op);
    } else {
      DefaultVisitAttrStmt(op);
    }
  }

  void VisitStmt_(const AssertStmtNode* op) override {
    if (f_visit_assert_stmt != nullptr) {
      f_visit_assert_stmt(op);
    } else {
      DefaultVisitAssertStmt(op);
    }
  }

  void VisitStmt_(const SeqStmtNode* op) override {
    if (f_visit_seq_stmt != nullptr) {
      f_visit_seq_stmt(op);
    } else {
      DefaultVisitSeqStmt(op);
    }
  }

  void VisitExpr_(const CallNode* op) override {
    if (f_visit_call != nullptr) {
      f_visit_call(op);
    } else {
      DefaultVisitCall(op);
    }
  }

  void VisitExpr_(const LetNode* op) override {
    if (f_visit_let != nullptr) {
      f_visit_let(op);
    } else {
      DefaultVisitLet(op);
    }
  }

  void VisitExpr_(const ReduceNode* op) override {
    if (f_visit_reduce != nullptr) {
      f_visit_reduce(op);
    } else {
      DefaultVisitReduce(op);
    }
  }

  void DefaultVisitFor(const ForNode* op) {
    constraint_scope_.WithNewScope([&]() {
      analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      StmtExprVisitor::VisitStmt_(op);
    });
  }

  void DefaultVisitSBlock(const SBlockNode* op) {
    constraint_scope_.WithNewScope([&]() {
      for (const IterVar& iter_var : op->iter_vars) {
        analyzer_->Bind(iter_var->var, iter_var->dom);
      }
      StmtExprVisitor::VisitStmt_(op);
    });
  }

  void DefaultVisitBind(const BindNode* op) {
    this->VisitExpr(op->value);
    if (SideEffect(op->value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, op->value);
    }
  }

  void DefaultVisitIfThenElse(const IfThenElseNode* op) {
    constraint_scope_.WithNewScope([&]() {
      this->VisitExpr(op->condition);
      PrimExpr real_condition = ExtractRealCondition(op->condition);
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_.get(), real_condition);
        this->VisitStmt(op->then_case);
      });
      if (op->else_case) {
        constraint_scope_.WithNewScope([&]() {
          constraint_scope_.Current().Emplace(analyzer_.get(),
                                              analyzer_->rewrite_simplify(Not(real_condition)));
          this->VisitStmt(op->else_case.value());
        });
      }
    });
  }

  void DefaultVisitAttrStmt(const AttrStmtNode* op) {
    constraint_scope_.WithNewScope([&]() {
      if (op->attr_key == tirx::attr::thread_extent ||
          op->attr_key == s_tir::attr::virtual_thread) {
        IterVar iv = Downcast<IterVar>(op->node);
        TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
        analyzer_->Bind(iv->var, Range::FromMinExtent(IntImm(op->value->dtype, 0), op->value));
      }
      StmtExprVisitor::VisitStmt_(op);
    });
  }

  void DefaultVisitAssertStmt(const AssertStmtNode* op) {
    this->VisitExpr(op->condition);
    constraint_scope_.Current().Emplace(analyzer_.get(), op->condition);
    this->VisitExpr(op->error_kind);
    for (const StringImm& message : op->message_parts) {
      this->VisitExpr(message);
    }
  }

  void DefaultVisitSeqStmt(const SeqStmtNode* op) { StmtExprVisitor::VisitStmt_(op); }

  void DefaultVisitCall(const CallNode* op) {
    static auto op_if_then_else = Op::Get("tirx.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      PrimExpr cond = op->args[0];
      this->VisitExpr(op->args[0]);
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_.get(), cond);
        this->VisitExpr(op->args[1]);
      });
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_.get(),
                                            analyzer_->rewrite_simplify(Not(cond)));
        this->VisitExpr(op->args[2]);
      });
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void DefaultVisitLet(const LetNode* op) {
    this->VisitExpr(op->value);
    if (SideEffect(op->value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, op->value);
    }
    this->VisitExpr(op->body);
  }

  void DefaultVisitReduce(const ReduceNode* op) {
    for (const IterVar& iv : op->axis) {
      analyzer_->Bind(iv->var, iv->dom);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  static FStmtType InitStmtVTable() {
    FStmtType vtable;
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(BindNode, DefaultVisitBind);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(AttrStmtNode, DefaultVisitAttrStmt);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(IfThenElseNode, DefaultVisitIfThenElse);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(ForNode, DefaultVisitFor);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(WhileNode);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(AllocBufferNode);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(DeclBufferNode);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(BufferStoreNode);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(AssertStmtNode, DefaultVisitAssertStmt);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(SeqStmtNode, DefaultVisitSeqStmt);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(EvaluateNode);
    PY_ANALYZER_STMT_VISITOR_DEFAULT_DISPATCH(SBlockNode, DefaultVisitSBlock);
    PY_ANALYZER_STMT_VISITOR_BASE_DISPATCH(SBlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }

  static FExprType InitExprVTable() {
    FExprType vtable;
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(VarNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(SizeVarNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(BufferLoadNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(ProducerLoadNode);
    PY_ANALYZER_EXPR_VISITOR_DEFAULT_DISPATCH(LetNode, DefaultVisitLet);
    PY_ANALYZER_EXPR_VISITOR_DEFAULT_DISPATCH(CallNode, DefaultVisitCall);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(AddNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(SubNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(MulNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(DivNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(ModNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(FloorDivNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(FloorModNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(MinNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(MaxNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(EQNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(NENode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(LTNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(LENode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(GTNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(GENode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(AndNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(OrNode);
    PY_ANALYZER_EXPR_VISITOR_DEFAULT_DISPATCH(ReduceNode, DefaultVisitReduce);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(CastNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(NotNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(SelectNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(RampNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(ShuffleNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(BroadcastNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(IntImmNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(FloatImmNode);
    PY_ANALYZER_EXPR_VISITOR_BASE_DISPATCH(StringImmNode);
    vtable.Finalize();
    return vtable;
  }
};

class PyStmtExprVisitorWithAnalyzer : public ffi::ObjectRef {
 public:
  explicit PyStmtExprVisitorWithAnalyzer(ffi::ObjectPtr<PyStmtExprVisitorWithAnalyzerNode> data)
      : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }

  TVM_DLL static PyStmtExprVisitorWithAnalyzer MakePyStmtExprVisitorWithAnalyzer(
      ffi::Array<ffi::Optional<ffi::Function>> callbacks) {
    ffi::ObjectPtr<PyStmtExprVisitorWithAnalyzerNode> n =
        ffi::make_object<PyStmtExprVisitorWithAnalyzerNode>();
    SetStmtExprFunctorCallbacks(n.get(), callbacks);
    return PyStmtExprVisitorWithAnalyzer(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyStmtExprVisitorWithAnalyzer, ffi::ObjectRef,
                                                PyStmtExprVisitorWithAnalyzerNode);
};

class PyStmtExprMutatorWithAnalyzerNode : public PyStmtExprMutatorNode {
 private:
  using TSelf = PyStmtExprMutatorWithAnalyzerNode;
  using FExprType = tvm::NodeFunctor<PrimExpr(const ffi::ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<Stmt(const ffi::ObjectRef& n, TSelf* self)>;

 public:
  PyStmtExprMutatorWithAnalyzerNode() = default;
  PyStmtExprMutatorWithAnalyzerNode(const PyStmtExprMutatorWithAnalyzerNode& other)
      : PyStmtExprMutatorNode(other) {}

  ffi::Function GetAnalyzer() { return MakeAnalyzerModule(analyzer_); }

  PrimExpr DefaultVisitExprWithAnalyzer(const PrimExpr& expr) {
    static FExprType vtable = InitExprVTable();
    return vtable(expr, this);
  }

  Stmt DefaultVisitStmtWithAnalyzer(const Stmt& stmt) {
    static FStmtType vtable = InitStmtVTable();
    return vtable(stmt, this);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PyStmtExprMutatorWithAnalyzerNode>();
  }

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.PyStmtExprMutatorWithAnalyzer",
                              PyStmtExprMutatorWithAnalyzerNode, PyStmtExprMutatorNode);

 private:
  std::shared_ptr<arith::Analyzer> analyzer_{std::make_shared<arith::Analyzer>()};
  ScopeStack<WithGroup<arith::ConstraintContext>> constraint_scope_;

  Stmt VisitStmt_(const ForNode* op) override {
    if (f_visit_for != nullptr) {
      return f_visit_for(op).cast<Stmt>();
    }
    return DefaultVisitFor(op);
  }

  Stmt VisitStmt_(const SBlockNode* op) override {
    if (f_visit_block != nullptr) {
      return f_visit_block(op).cast<Stmt>();
    }
    return DefaultVisitSBlock(op);
  }

  Stmt VisitStmt_(const BindNode* op) override {
    if (f_visit_bind != nullptr) {
      return f_visit_bind(op).cast<Stmt>();
    }
    return DefaultVisitBind(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (f_visit_if_then_else != nullptr) {
      return f_visit_if_then_else(op).cast<Stmt>();
    }
    return DefaultVisitIfThenElse(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (f_visit_attr_stmt != nullptr) {
      return f_visit_attr_stmt(op).cast<Stmt>();
    }
    return DefaultVisitAttrStmt(op);
  }

  Stmt VisitStmt_(const AssertStmtNode* op) override {
    if (f_visit_assert_stmt != nullptr) {
      return f_visit_assert_stmt(op).cast<Stmt>();
    }
    return DefaultVisitAssertStmt(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    if (f_visit_seq_stmt != nullptr) {
      return f_visit_seq_stmt(op).cast<Stmt>();
    }
    return DefaultVisitSeqStmt(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
    if (f_visit_call != nullptr) {
      return f_visit_call(op).cast<PrimExpr>();
    }
    return DefaultVisitCall(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) override {
    if (f_visit_let != nullptr) {
      return f_visit_let(op).cast<PrimExpr>();
    }
    return DefaultVisitLet(op);
  }

  PrimExpr VisitExpr_(const SelectNode* op) override {
    if (f_visit_select != nullptr) {
      return f_visit_select(op).cast<PrimExpr>();
    }
    return DefaultVisitSelect(op);
  }

  PrimExpr VisitExpr_(const ReduceNode* op) override {
    if (f_visit_reduce != nullptr) {
      return f_visit_reduce(op).cast<PrimExpr>();
    }
    return DefaultVisitReduce(op);
  }

  Stmt DefaultVisitFor(const ForNode* op) {
    return constraint_scope_.WithNewScope([&]() -> Stmt {
      Range dom = Range::FromMinExtent(op->min, op->extent);
      analyzer_->Bind(op->loop_var, dom);
      return StmtExprMutator::VisitStmt_(op);
    });
  }

  Stmt DefaultVisitSBlock(const SBlockNode* op) {
    return constraint_scope_.WithNewScope([&]() -> Stmt {
      for (const IterVar& iter_var : op->iter_vars) {
        analyzer_->Bind(iter_var->var, iter_var->dom);
      }
      return StmtExprMutator::VisitStmt_(op);
    });
  }

  Stmt DefaultVisitBind(const BindNode* op) {
    PrimExpr value = this->VisitExpr(op->value);
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
    }
    if (value.same_as(op->value)) {
      return ffi::GetRef<Stmt>(op);
    }
    auto n = this->CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }

  Stmt DefaultVisitIfThenElse(const IfThenElseNode* op) {
    return constraint_scope_.WithNewScope([&]() -> Stmt {
      PrimExpr condition = this->VisitExpr(op->condition);
      PrimExpr real_condition = ExtractRealCondition(condition);
      Stmt then_case;
      ffi::Optional<Stmt> else_case;
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_.get(), real_condition);
        then_case = this->VisitStmt(op->then_case);
      });
      if (op->else_case) {
        PrimExpr neg_condition = analyzer_->rewrite_simplify(Not(real_condition));
        constraint_scope_.WithNewScope([&]() {
          constraint_scope_.Current().Emplace(analyzer_.get(), neg_condition);
          else_case = this->VisitStmt(op->else_case.value());
        });
      }
      if (is_one(real_condition)) return then_case;
      if (is_zero(real_condition)) return else_case.value_or(Evaluate(0));
      if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return ffi::GetRef<Stmt>(op);
      }
      auto n = this->CopyOnWrite(op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      return Stmt(n);
    });
  }

  Stmt DefaultVisitAttrStmt(const AttrStmtNode* op) {
    return constraint_scope_.WithNewScope([&]() -> Stmt {
      if (op->attr_key == tirx::attr::thread_extent ||
          op->attr_key == s_tir::attr::virtual_thread) {
        IterVar iv = Downcast<IterVar>(op->node);
        TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
        analyzer_->Bind(iv->var, Range::FromMinExtent(make_zero(op->value.dtype()), op->value));
      }
      return StmtExprMutator::VisitStmt_(op);
    });
  }

  Stmt DefaultVisitAssertStmt(const AssertStmtNode* op) {
    PrimExpr condition = this->VisitExpr(op->condition);
    constraint_scope_.Current().Emplace(analyzer_.get(), condition);
    PrimExpr error_kind = this->VisitExpr(op->error_kind);
    ffi::Array<StringImm> message_parts;
    bool message_parts_same = true;
    for (const StringImm& message : op->message_parts) {
      StringImm new_message = Downcast<StringImm>(this->VisitExpr(message));
      if (!new_message.same_as(message)) {
        message_parts_same = false;
      }
      message_parts.push_back(std::move(new_message));
    }
    if (condition.same_as(op->condition) && error_kind.same_as(op->error_kind) &&
        message_parts_same) {
      return ffi::GetRef<Stmt>(op);
    }
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->error_kind = Downcast<StringImm>(std::move(error_kind));
    if (!message_parts_same) {
      n->message_parts = std::move(message_parts);
    }
    return Stmt(n);
  }

  Stmt DefaultVisitSeqStmt(const SeqStmtNode* op) { return StmtExprMutator::VisitStmt_(op); }

  PrimExpr DefaultVisitCall(const CallNode* op) {
    static auto op_if_then_else = Op::Get("tirx.if_then_else");
    if (op->op.same_as(op_if_then_else)) {
      PrimExpr cond = this->VisitExpr(op->args[0]);
      PrimExpr true_value;
      PrimExpr false_value;
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_.get(), cond);
        true_value = this->VisitExpr(op->args[1]);
      });
      constraint_scope_.WithNewScope([&]() {
        PrimExpr not_cond = analyzer_->rewrite_simplify(Not(cond));
        constraint_scope_.Current().Emplace(analyzer_.get(), not_cond);
        false_value = this->VisitExpr(op->args[2]);
      });
      if (is_zero(cond)) return false_value;
      if (is_one(cond)) return true_value;
      if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
          false_value.same_as(op->args[2])) {
        return ffi::GetRef<PrimExpr>(op);
      }
      return Call(op->dtype, op->op, {cond, true_value, false_value}, op->annotations, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr DefaultVisitLet(const LetNode* op) {
    PrimExpr value = this->VisitExpr(op->value);
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
    }
    PrimExpr body = this->VisitExpr(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Let(op->var, value, body);
  }

  PrimExpr DefaultVisitSelect(const SelectNode* op) {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr true_value;
    PrimExpr false_value;
    constraint_scope_.WithNewScope([&]() {
      constraint_scope_.Current().Emplace(analyzer_.get(), cond);
      true_value = this->VisitExpr(op->true_value);
    });
    constraint_scope_.WithNewScope([&]() {
      PrimExpr neg_cond = analyzer_->rewrite_simplify(Not(cond));
      constraint_scope_.Current().Emplace(analyzer_.get(), neg_cond);
      false_value = this->VisitExpr(op->false_value);
    });
    if (is_zero(cond)) return false_value;
    if (is_one(cond)) return true_value;
    if (cond.same_as(op->condition) && true_value.same_as(op->true_value) &&
        false_value.same_as(op->false_value)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Select(cond, true_value, false_value);
  }

  PrimExpr DefaultVisitReduce(const ReduceNode* op) {
    for (const IterVar& iv : op->axis) {
      analyzer_->Bind(iv->var, iv->dom);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  static FStmtType InitStmtVTable() {
    FStmtType vtable;
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(BindNode, DefaultVisitBind);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(AttrStmtNode, DefaultVisitAttrStmt);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(IfThenElseNode, DefaultVisitIfThenElse);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(ForNode, DefaultVisitFor);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(WhileNode);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(AllocBufferNode);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(DeclBufferNode);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(BufferStoreNode);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(AssertStmtNode, DefaultVisitAssertStmt);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(SeqStmtNode, DefaultVisitSeqStmt);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(EvaluateNode);
    PY_ANALYZER_STMT_MUTATOR_DEFAULT_DISPATCH(SBlockNode, DefaultVisitSBlock);
    PY_ANALYZER_STMT_MUTATOR_BASE_DISPATCH(SBlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }

  static FExprType InitExprVTable() {
    FExprType vtable;
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(VarNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(SizeVarNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(BufferLoadNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(ProducerLoadNode);
    PY_ANALYZER_EXPR_MUTATOR_DEFAULT_DISPATCH(LetNode, DefaultVisitLet);
    PY_ANALYZER_EXPR_MUTATOR_DEFAULT_DISPATCH(CallNode, DefaultVisitCall);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(AddNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(SubNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(MulNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(DivNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(ModNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(FloorDivNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(FloorModNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(MinNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(MaxNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(EQNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(NENode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(LTNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(LENode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(GTNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(GENode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(AndNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(OrNode);
    PY_ANALYZER_EXPR_MUTATOR_DEFAULT_DISPATCH(ReduceNode, DefaultVisitReduce);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(CastNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(NotNode);
    PY_ANALYZER_EXPR_MUTATOR_DEFAULT_DISPATCH(SelectNode, DefaultVisitSelect);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(RampNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(ShuffleNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(BroadcastNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(IntImmNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(FloatImmNode);
    PY_ANALYZER_EXPR_MUTATOR_BASE_DISPATCH(StringImmNode);
    vtable.Finalize();
    return vtable;
  }
};

class PyStmtExprMutatorWithAnalyzer : public ffi::ObjectRef {
 public:
  explicit PyStmtExprMutatorWithAnalyzer(ffi::ObjectPtr<PyStmtExprMutatorWithAnalyzerNode> data)
      : ffi::ObjectRef(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }

  TVM_DLL static PyStmtExprMutatorWithAnalyzer MakePyStmtExprMutatorWithAnalyzer(
      ffi::Array<ffi::Optional<ffi::Function>> callbacks) {
    ffi::ObjectPtr<PyStmtExprMutatorWithAnalyzerNode> n =
        ffi::make_object<PyStmtExprMutatorWithAnalyzerNode>();
    SetStmtExprFunctorCallbacks(n.get(), callbacks);
    return PyStmtExprMutatorWithAnalyzer(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PyStmtExprMutatorWithAnalyzer, ffi::ObjectRef,
                                                PyStmtExprMutatorWithAnalyzerNode);
};

// ================================================
// TVM Register
// ================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  PyStmtExprVisitorNode::RegisterReflection();
  PyStmtExprMutatorNode::RegisterReflection();
  PyStmtExprVisitorWithAnalyzerNode::RegisterReflection();
  PyStmtExprMutatorWithAnalyzerNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.MakePyStmtExprVisitor", PyStmtExprVisitor::MakePyStmtExprVisitor)
      .def("tirx.MakePyStmtExprMutator", PyStmtExprMutator::MakePyStmtExprMutator)
      .def("tirx.MakePyStmtExprVisitorWithAnalyzer",
           PyStmtExprVisitorWithAnalyzer::MakePyStmtExprVisitorWithAnalyzer)
      .def("tirx.MakePyStmtExprMutatorWithAnalyzer",
           PyStmtExprMutatorWithAnalyzer::MakePyStmtExprMutatorWithAnalyzer);
}

// StmtExprVisitor
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PyStmtExprVisitorDefaultVisitExpr",
           [](PyStmtExprVisitor visitor, const PrimExpr& expr) { visitor->DefaultVisitExpr(expr); })
      .def("tirx.PyStmtExprVisitorDefaultVisitStmt",
           [](PyStmtExprVisitor visitor, const Stmt& stmt) { visitor->DefaultVisitStmt(stmt); })
      .def("tirx.PyStmtExprVisitorVisitStmt",
           [](PyStmtExprVisitor visitor, const Stmt& stmt) { visitor->VisitStmt(stmt); })
      .def("tirx.PyStmtExprVisitorVisitExpr",
           [](PyStmtExprVisitor visitor, const PrimExpr& expr) { visitor->VisitExpr(expr); });
}

// StmtExprVisitorWithAnalyzer
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PyStmtExprVisitorWithAnalyzerDefaultVisitExpr",
           [](PyStmtExprVisitorWithAnalyzer visitor, const PrimExpr& expr) {
             visitor->DefaultVisitExprWithAnalyzer(expr);
           })
      .def("tirx.PyStmtExprVisitorWithAnalyzerDefaultVisitStmt",
           [](PyStmtExprVisitorWithAnalyzer visitor, const Stmt& stmt) {
             visitor->DefaultVisitStmtWithAnalyzer(stmt);
           })
      .def(
          "tirx.PyStmtExprVisitorWithAnalyzerVisitStmt",
          [](PyStmtExprVisitorWithAnalyzer visitor, const Stmt& stmt) { visitor->VisitStmt(stmt); })
      .def("tirx.PyStmtExprVisitorWithAnalyzerVisitExpr",
           [](PyStmtExprVisitorWithAnalyzer visitor, const PrimExpr& expr) {
             visitor->VisitExpr(expr);
           })
      .def("tirx.PyStmtExprVisitorWithAnalyzerGetAnalyzer",
           [](PyStmtExprVisitorWithAnalyzer visitor) { return visitor->GetAnalyzer(); });
}

// StmtExprMutator
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PyStmtExprMutatorDefaultVisitExpr",
           [](PyStmtExprMutator mutator, const PrimExpr& expr) {
             return mutator->DefaultVisitExpr(expr);
           })
      .def("tirx.PyStmtExprMutatorDefaultVisitStmt",
           [](PyStmtExprMutator mutator, const Stmt& stmt) {
             return mutator->DefaultVisitStmt(stmt);
           })
      .def("tirx.PyStmtExprMutatorVisitExpr",
           [](PyStmtExprMutator mutator, const PrimExpr& expr) { return mutator->VisitExpr(expr); })
      .def("tirx.PyStmtExprMutatorVisitStmt",
           [](PyStmtExprMutator mutator, const Stmt& stmt) { return mutator->VisitStmt(stmt); });
}

// StmtExprMutatorWithAnalyzer
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PyStmtExprMutatorWithAnalyzerDefaultVisitExpr",
           [](PyStmtExprMutatorWithAnalyzer mutator, const PrimExpr& expr) {
             return mutator->DefaultVisitExprWithAnalyzer(expr);
           })
      .def("tirx.PyStmtExprMutatorWithAnalyzerDefaultVisitStmt",
           [](PyStmtExprMutatorWithAnalyzer mutator, const Stmt& stmt) {
             return mutator->DefaultVisitStmtWithAnalyzer(stmt);
           })
      .def("tirx.PyStmtExprMutatorWithAnalyzerVisitExpr",
           [](PyStmtExprMutatorWithAnalyzer mutator, const PrimExpr& expr) {
             return mutator->VisitExpr(expr);
           })
      .def("tirx.PyStmtExprMutatorWithAnalyzerVisitStmt",
           [](PyStmtExprMutatorWithAnalyzer mutator, const Stmt& stmt) {
             return mutator->VisitStmt(stmt);
           })
      .def("tirx.PyStmtExprMutatorWithAnalyzerGetAnalyzer",
           [](PyStmtExprMutatorWithAnalyzer mutator) { return mutator->GetAnalyzer(); });
}

}  // namespace tirx
}  // namespace tvm
