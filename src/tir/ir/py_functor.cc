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
 * \file src/tir/ir/py_functor.cc
 * \brief The python interface of ExprVisitor/ExprMutator, StmtVisitor/StmtMutator,
 *        StmtExprVisitor/StmtExprMutator.
 */

#include <tvm/ffi/reflection/reflection.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

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

#define IR_EXPR_VISITOR_DEFAULT_DISPATCH(OP)                             \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitExpr_(static_cast<const OP*>(n.get()));  \
  });

#define PY_STMT_VISITOR_DISPATCH(OP, PY_FUNC) \
  void VisitStmt_(const OP* op) override {    \
    if (PY_FUNC != nullptr) {                 \
      PY_FUNC(op);                            \
    } else {                                  \
      StmtExprVisitor::VisitStmt_(op);        \
    }                                         \
  }

#define PY_STMT_VISITOR_DEFAULT_DISPATCH(OP)                             \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    self->StmtExprVisitor::VisitStmt_(static_cast<const OP*>(n.get()));  \
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
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) {       \
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
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) {       \
    return self->StmtExprMutator::VisitStmt_(static_cast<const OP*>(n.get())); \
  });

/*! \brief The python interface of StmtExprVisitor. */
class PyStmtExprVisitorNode : public Object, public StmtExprVisitor {
 private:
  using TSelf = PyStmtExprVisitorNode;
  using FExprType = tvm::NodeFunctor<void(const ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<void(const ObjectRef& n, TSelf* self)>;

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
  /*! \brief The packed function to the `VisitStmt_(const LetStmtNode* op)` function. */
  ffi::Function f_visit_attr_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const IfThenElseNode* op)` function. */
  ffi::Function f_visit_if_then_else{nullptr};  // NOLINT(readability/braces)
  /*! \brief The packed function to the `VisitStmt_(const ForNode* op)` function. */
  ffi::Function f_visit_let_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AttrStmtNode* op)` function. */
  ffi::Function f_visit_for{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const WhileNode* op)` function. */
  ffi::Function f_visit_while{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocateNode* op)` function. */
  ffi::Function f_visit_allocate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocateConstNode* op)` function. */
  ffi::Function f_visit_allocate_const{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const DeclBufferNode* op)` function. */
  ffi::Function f_visit_decl_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferStoreNode* op)` function. */
  ffi::Function f_visit_buffer_store{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferRealizeNode* op)` function. */
  ffi::Function f_visit_buffer_realize{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AssertStmtNode* op)` function. */
  ffi::Function f_visit_assert_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SeqStmtNode* op)` function. */
  ffi::Function f_visit_seq_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const EvaluateNode* op)` function. */
  ffi::Function f_visit_evaluate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BlockNode* op)` function. */
  ffi::Function f_visit_block{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BlockRealizeNode* op)` function. */
  ffi::Function f_visit_block_realize{nullptr};

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
    // No fields to register as they are not visited
  }

  static constexpr const char* _type_key = "tir.PyStmtExprVisitor";
  TVM_DECLARE_BASE_OBJECT_INFO(PyStmtExprVisitorNode, Object);

 private:
  // Statement functions
  PY_STMT_VISITOR_DISPATCH(LetStmtNode, f_visit_let_stmt);
  PY_STMT_VISITOR_DISPATCH(AttrStmtNode, f_visit_attr_stmt);
  PY_STMT_VISITOR_DISPATCH(IfThenElseNode, f_visit_if_then_else);
  PY_STMT_VISITOR_DISPATCH(ForNode, f_visit_for);
  PY_STMT_VISITOR_DISPATCH(WhileNode, f_visit_while);
  PY_STMT_VISITOR_DISPATCH(AllocateNode, f_visit_allocate);
  PY_STMT_VISITOR_DISPATCH(AllocateConstNode, f_visit_allocate_const);
  PY_STMT_VISITOR_DISPATCH(DeclBufferNode, f_visit_decl_buffer);
  PY_STMT_VISITOR_DISPATCH(BufferStoreNode, f_visit_buffer_store);
  PY_STMT_VISITOR_DISPATCH(BufferRealizeNode, f_visit_buffer_realize);
  PY_STMT_VISITOR_DISPATCH(AssertStmtNode, f_visit_assert_stmt);
  PY_STMT_VISITOR_DISPATCH(SeqStmtNode, f_visit_seq_stmt);
  PY_STMT_VISITOR_DISPATCH(EvaluateNode, f_visit_evaluate);
  PY_STMT_VISITOR_DISPATCH(BlockNode, f_visit_block);
  PY_STMT_VISITOR_DISPATCH(BlockRealizeNode, f_visit_block_realize);
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
    PY_STMT_VISITOR_DEFAULT_DISPATCH(LetStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AttrStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(IfThenElseNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(ForNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(WhileNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AllocateNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AllocateConstNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(DeclBufferNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BufferStoreNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BufferRealizeNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(AssertStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(SeqStmtNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(EvaluateNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BlockNode);
    PY_STMT_VISITOR_DEFAULT_DISPATCH(BlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief Managed reference to PyStmtExprVisitorNode.
 * \sa PyStmtExprVisitorNode
 */
class PyStmtExprVisitor : public ObjectRef {
 public:
  TVM_DLL static PyStmtExprVisitor MakePyStmtExprVisitor(ffi::Function f_visit_stmt,            //
                                                         ffi::Function f_visit_expr,            //
                                                         ffi::Function f_visit_let_stmt,        //
                                                         ffi::Function f_visit_attr_stmt,       //
                                                         ffi::Function f_visit_if_then_else,    //
                                                         ffi::Function f_visit_for,             //
                                                         ffi::Function f_visit_while,           //
                                                         ffi::Function f_visit_allocate,        //
                                                         ffi::Function f_visit_allocate_const,  //
                                                         ffi::Function f_visit_decl_buffer,     //
                                                         ffi::Function f_visit_buffer_store,    //
                                                         ffi::Function f_visit_buffer_realize,  //
                                                         ffi::Function f_visit_assert_stmt,     //
                                                         ffi::Function f_visit_seq_stmt,        //
                                                         ffi::Function f_visit_evaluate,        //
                                                         ffi::Function f_visit_block,           //
                                                         ffi::Function f_visit_block_realize,   //
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
    ObjectPtr<PyStmtExprVisitorNode> n = make_object<PyStmtExprVisitorNode>();
    n->f_visit_stmt = std::move(f_visit_stmt);
    n->f_visit_expr = std::move(f_visit_expr);
    // Set statement functions
    n->f_visit_let_stmt = std::move(f_visit_let_stmt);
    n->f_visit_attr_stmt = std::move(f_visit_attr_stmt);
    n->f_visit_if_then_else = std::move(f_visit_if_then_else);
    n->f_visit_for = std::move(f_visit_for);
    n->f_visit_while = std::move(f_visit_while);
    n->f_visit_allocate = std::move(f_visit_allocate);
    n->f_visit_allocate_const = std::move(f_visit_allocate_const);
    n->f_visit_decl_buffer = std::move(f_visit_decl_buffer);
    n->f_visit_buffer_store = std::move(f_visit_buffer_store);
    n->f_visit_buffer_realize = std::move(f_visit_buffer_realize);
    n->f_visit_assert_stmt = std::move(f_visit_assert_stmt);
    n->f_visit_seq_stmt = std::move(f_visit_seq_stmt);
    n->f_visit_evaluate = std::move(f_visit_evaluate);
    n->f_visit_block = std::move(f_visit_block);
    n->f_visit_block_realize = std::move(f_visit_block_realize);
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

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyStmtExprVisitor, ObjectRef,
                                                    PyStmtExprVisitorNode);
};

/*! \brief The python interface of StmtExprMutator. */
class PyStmtExprMutatorNode : public Object, public StmtExprMutator {
 private:
  using TSelf = PyStmtExprMutatorNode;
  using FExprType = tvm::NodeFunctor<PrimExpr(const ObjectRef& n, TSelf* self)>;
  using FStmtType = tvm::NodeFunctor<Stmt(const ObjectRef& n, TSelf* self)>;

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
  /*! \brief The packed function to the `VisitStmt_(const LetStmtNode* op)` function. */
  ffi::Function f_visit_let_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AttrStmtNode* op)` function. */
  ffi::Function f_visit_attr_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const IfThenElseNode* op)` function. */
  ffi::Function f_visit_if_then_else{nullptr};  // NOLINT(readability/braces)
  /*! \brief The packed function to the `VisitStmt_(const ForNode* op)` function. */
  ffi::Function f_visit_for{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const WhileNode* op)` function. */
  ffi::Function f_visit_while{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocateNode* op)` function. */
  ffi::Function f_visit_allocate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AllocateConstNode* op)` function. */
  ffi::Function f_visit_allocate_const{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const DeclBufferNode* op)` function. */
  ffi::Function f_visit_decl_buffer{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferStoreNode* op)` function. */
  ffi::Function f_visit_buffer_store{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BufferRealizeNode* op)` function. */
  ffi::Function f_visit_buffer_realize{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const AssertStmtNode* op)` function. */
  ffi::Function f_visit_assert_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const SeqStmtNode* op)` function. */
  ffi::Function f_visit_seq_stmt{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const EvaluateNode* op)` function. */
  ffi::Function f_visit_evaluate{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BlockNode* op)` function. */
  ffi::Function f_visit_block{nullptr};
  /*! \brief The packed function to the `VisitStmt_(const BlockRealizeNode* op)` function. */
  ffi::Function f_visit_block_realize{nullptr};

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
    // No fields to register as they are not visited
  }

  static constexpr const char* _type_key = "tir.PyStmtExprMutator";
  TVM_DECLARE_BASE_OBJECT_INFO(PyStmtExprMutatorNode, Object);

 private:
  // Statement functions
  PY_STMT_MUTATOR_DISPATCH(LetStmtNode, f_visit_let_stmt);
  PY_STMT_MUTATOR_DISPATCH(AttrStmtNode, f_visit_attr_stmt);
  PY_STMT_MUTATOR_DISPATCH(IfThenElseNode, f_visit_if_then_else);
  PY_STMT_MUTATOR_DISPATCH(ForNode, f_visit_for);
  PY_STMT_MUTATOR_DISPATCH(WhileNode, f_visit_while);
  PY_STMT_MUTATOR_DISPATCH(AllocateNode, f_visit_allocate);
  PY_STMT_MUTATOR_DISPATCH(AllocateConstNode, f_visit_allocate_const);
  PY_STMT_MUTATOR_DISPATCH(DeclBufferNode, f_visit_decl_buffer);
  PY_STMT_MUTATOR_DISPATCH(BufferStoreNode, f_visit_buffer_store);
  PY_STMT_MUTATOR_DISPATCH(BufferRealizeNode, f_visit_buffer_realize);
  PY_STMT_MUTATOR_DISPATCH(AssertStmtNode, f_visit_assert_stmt);
  PY_STMT_MUTATOR_DISPATCH(SeqStmtNode, f_visit_seq_stmt);
  PY_STMT_MUTATOR_DISPATCH(EvaluateNode, f_visit_evaluate);
  PY_STMT_MUTATOR_DISPATCH(BlockNode, f_visit_block);
  PY_STMT_MUTATOR_DISPATCH(BlockRealizeNode, f_visit_block_realize);
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
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(LetStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AttrStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(IfThenElseNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(ForNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(WhileNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AllocateNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AllocateConstNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(DeclBufferNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BufferStoreNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BufferRealizeNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(AssertStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(SeqStmtNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(EvaluateNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BlockNode);
    PY_STMT_MUTATOR_DEFAULT_DISPATCH(BlockRealizeNode);
    vtable.Finalize();
    return vtable;
  }
};

/*! \brief Managed reference to PyStmtExprMutatorNode. */
class PyStmtExprMutator : public ObjectRef {
 public:
  /*!
   * \brief Create a PyStmtExprMutator with customized methods on the python-side.
   * \return The PyStmtExprMutator created.
   */
  TVM_DLL static PyStmtExprMutator MakePyStmtExprMutator(ffi::Function f_visit_stmt,            //
                                                         ffi::Function f_visit_expr,            //
                                                         ffi::Function f_visit_let_stmt,        //
                                                         ffi::Function f_visit_attr_stmt,       //
                                                         ffi::Function f_visit_if_then_else,    //
                                                         ffi::Function f_visit_for,             //
                                                         ffi::Function f_visit_while,           //
                                                         ffi::Function f_visit_allocate,        //
                                                         ffi::Function f_visit_allocate_const,  //
                                                         ffi::Function f_visit_decl_buffer,     //
                                                         ffi::Function f_visit_buffer_store,    //
                                                         ffi::Function f_visit_buffer_realize,  //
                                                         ffi::Function f_visit_assert_stmt,     //
                                                         ffi::Function f_visit_seq_stmt,        //
                                                         ffi::Function f_visit_evaluate,        //
                                                         ffi::Function f_visit_block,           //
                                                         ffi::Function f_visit_block_realize,   //
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
    ObjectPtr<PyStmtExprMutatorNode> n = make_object<PyStmtExprMutatorNode>();
    n->f_visit_stmt = std::move(f_visit_stmt);
    n->f_visit_expr = std::move(f_visit_expr);
    // Statement functions
    n->f_visit_let_stmt = std::move(f_visit_let_stmt);
    n->f_visit_attr_stmt = std::move(f_visit_attr_stmt);
    n->f_visit_if_then_else = std::move(f_visit_if_then_else);
    n->f_visit_for = std::move(f_visit_for);
    n->f_visit_while = std::move(f_visit_while);
    n->f_visit_allocate = std::move(f_visit_allocate);
    n->f_visit_allocate_const = std::move(f_visit_allocate_const);
    n->f_visit_decl_buffer = std::move(f_visit_decl_buffer);
    n->f_visit_buffer_store = std::move(f_visit_buffer_store);
    n->f_visit_buffer_realize = std::move(f_visit_buffer_realize);
    n->f_visit_assert_stmt = std::move(f_visit_assert_stmt);
    n->f_visit_seq_stmt = std::move(f_visit_seq_stmt);
    n->f_visit_evaluate = std::move(f_visit_evaluate);
    n->f_visit_block = std::move(f_visit_block);
    n->f_visit_block_realize = std::move(f_visit_block_realize);
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

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PyStmtExprMutator, ObjectRef,
                                                    PyStmtExprMutatorNode);
};

// ================================================
// TVM Register
// ================================================

TVM_FFI_STATIC_INIT_BLOCK({
  PyStmtExprVisitorNode::RegisterReflection();
  PyStmtExprMutatorNode::RegisterReflection();
});

TVM_REGISTER_NODE_TYPE(PyStmtExprVisitorNode);
TVM_REGISTER_NODE_TYPE(PyStmtExprMutatorNode);

TVM_FFI_REGISTER_GLOBAL("tir.MakePyStmtExprVisitor")
    .set_body_typed(PyStmtExprVisitor::MakePyStmtExprVisitor);
TVM_FFI_REGISTER_GLOBAL("tir.MakePyStmtExprMutator")
    .set_body_typed(PyStmtExprMutator::MakePyStmtExprMutator);

// StmtExprVisitor
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprVisitorDefaultVisitExpr")
    .set_body_typed([](PyStmtExprVisitor visitor, const PrimExpr& expr) {
      visitor->DefaultVisitExpr(expr);
    });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprVisitorDefaultVisitStmt")
    .set_body_typed([](PyStmtExprVisitor visitor, const Stmt& stmt) {
      visitor->DefaultVisitStmt(stmt);
    });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprVisitorVisitStmt")
    .set_body_typed([](PyStmtExprVisitor visitor, const Stmt& stmt) { visitor->VisitStmt(stmt); });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprVisitorVisitExpr")
    .set_body_typed([](PyStmtExprVisitor visitor, const PrimExpr& expr) {
      visitor->VisitExpr(expr);
    });

// StmtExprMutator
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprMutatorDefaultVisitExpr")
    .set_body_typed([](PyStmtExprMutator mutator, const PrimExpr& expr) {
      return mutator->DefaultVisitExpr(expr);
    });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprMutatorDefaultVisitStmt")
    .set_body_typed([](PyStmtExprMutator mutator, const Stmt& stmt) {
      return mutator->DefaultVisitStmt(stmt);
    });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprMutatorVisitExpr")
    .set_body_typed([](PyStmtExprMutator mutator, const PrimExpr& expr) {
      return mutator->VisitExpr(expr);
    });
TVM_FFI_REGISTER_GLOBAL("tir.PyStmtExprMutatorVisitStmt")
    .set_body_typed([](PyStmtExprMutator mutator, const Stmt& stmt) {
      return mutator->VisitStmt(stmt);
    });

}  // namespace tir
}  // namespace tvm
