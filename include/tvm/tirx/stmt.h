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
 * \file tvm/tirx/stmt.h
 * \brief TIR statements.
 */
// Acknowledgement: Many low-level stmts originate from Halide.
#ifndef TVM_TIRX_STMT_H_
#define TVM_TIRX_STMT_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace tirx {

/*! \brief Base node of all statements. */
class StmtNode : public ffi::Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  StmtNode() = default;
  explicit StmtNode(Span span) : span(span) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtNode>().def_ro("span", &StmtNode::span,
                                       refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;

  static constexpr const uint32_t _type_child_slots = 15;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.Stmt", StmtNode, ffi::Object);
};

/*! \brief Container of all statements */
class Stmt : public ffi::ObjectRef {
 public:
  explicit Stmt(ffi::ObjectPtr<StmtNode> node) : ffi::ObjectRef(std::move(node)) {
    TVM_FFI_CHECK(defined(), ValueError) << "Stmt expects a defined node";
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Stmt, ffi::ObjectRef, StmtNode);
};

/*!
 * \brief The container of seq statement.
 *        Represent a sequence of statements.
 */
class SeqStmtNode : public StmtNode {
 public:
  /*! \brief internal sequence content. */
  ffi::Array<Stmt> seq;

  /*! \return get the size of the sequence */
  size_t size() const { return seq.size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return seq[index]; }

  TVM_DLL static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.SeqStmt", SeqStmtNode, StmtNode);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   * \param span The location of this object in the source code.
   */
  TVM_DLL explicit SeqStmt(ffi::Array<Stmt> seq, Span span = Span());
  /*! \brief Wrap a statement, reusing an existing sequence when possible. */
  TVM_DLL SeqStmt(Stmt stmt, Span span = Span());
  SeqStmt(std::initializer_list<Stmt> seq, Span span = Span())
      : SeqStmt(ffi::Array<Stmt>(seq), std::move(span)) {}

  /*! \return get the size of the sequence */
  size_t size() const { return operator->()->size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
  explicit SeqStmt(ffi::ObjectPtr<SeqStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SeqStmt, Stmt, SeqStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtNode);
};

}  // namespace tirx
namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<tirx::SeqStmt> = false;

template <>
struct TypeTraits<tirx::SeqStmt>
    : public ObjectRefWithFallbackTraitsBase<tirx::SeqStmt, tirx::Stmt, Array<tirx::Stmt>> {
  static tirx::SeqStmt ConvertFallbackValue(tirx::Stmt stmt) {
    return tirx::SeqStmt(std::move(stmt));
  }
  static tirx::SeqStmt ConvertFallbackValue(Array<tirx::Stmt> seq) {
    return tirx::SeqStmt(std::move(seq));
  }
};
}  // namespace ffi
namespace tirx {

/*!
 * \brief Bind a variable to a value in the enclosing scope.
 *
 * BindNode has no body field. The bound variable is visible
 * in all subsequent statements within the same enclosing scope (SeqStmt,
 * ForNode.body, etc.). This enables flat (non-nested) IR sequences.
 */
class BindNode : public StmtNode {
 public:
  explicit BindNode(ffi::UnsafeInit tag) : var(tag), value(tag) {}

  BindNode(Var var, Expr value) : var(std::move(var)), value(std::move(value)) {}

  /*! \brief The variable being bound. */
  Var var;
  /*! \brief The value to bind to the variable. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BindNode>()
        .def_ro("var", &BindNode::var, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("value", &BindNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Bind", BindNode, StmtNode);
};

/*!
 * \brief Managed reference to BindNode.
 * \sa BindNode
 */
class Bind : public Stmt {
 public:
  TVM_DLL Bind(Var var, Expr value, Span span = Span());

  explicit Bind(ffi::ObjectPtr<BindNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Bind, Stmt, BindNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BindNode);
};

/*!
 * \brief Check FRegionGetBodyParams registration without invoking it or creating variables.
 */
TVM_DLL bool IsRegionOp(const Op& op);

/*!
 * \brief Validate region operands and construct fresh typed body parameters.
 * \param op The region operation.
 * \param args Operands evaluated in the enclosing scope.
 * \param attrs Attributes evaluated in the enclosing scope.
 * \return Parameters from the required FRegionGetBodyParams hook.
 * \throws ValueError if the operation has no registered region hook.
 */
TVM_DLL ffi::Array<Var> GetRegionBodyParams(Op op, ffi::Array<Expr> args, DictAttrs attrs);

/*!
 * \brief A single-body statement whose semantics are defined by an operator.
 * Operands are evaluated in the enclosing scope. Body parameters are definitions
 * at body entry; result variables are definitions following the region.
 */
class RegionStmtNode : public StmtNode {
 public:
  /*! \brief Operator defining the region's semantics. */
  Op op;
  /*! \brief Operands evaluated in the enclosing scope. */
  ffi::Array<Expr> args;
  /*! \brief Variables defined at body entry, visible only within the body. */
  ffi::Array<Var> body_params;
  /*! \brief Attributes evaluated in the enclosing scope. */
  DictAttrs attrs;
  /*! \brief Body evaluated with the body parameters in scope. */
  SeqStmt body;
  /*! \brief Variables defined after the region in the enclosing sequence. */
  ffi::Array<Var> result_vars;

  explicit RegionStmtNode(ffi::UnsafeInit tag) : op(tag), body(tag) {}

  RegionStmtNode(Op op, SeqStmt body) : op(std::move(op)), body(std::move(body)) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RegionStmtNode>()
        .def_ro("op", &RegionStmtNode::op)
        .def_ro("attrs", &RegionStmtNode::attrs)
        .def_ro("args", &RegionStmtNode::args)
        .def_ro("body_params", &RegionStmtNode::body_params,
                refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("body", &RegionStmtNode::body)
        .def_ro("result_vars", &RegionStmtNode::result_vars,
                refl::AttachFieldFlag::SEqHashDefSimple());
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.RegionStmt", RegionStmtNode, StmtNode);
};

/*! \brief Managed reference to RegionStmtNode. */
class RegionStmt : public Stmt {
 public:
  TVM_DLL RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params, DictAttrs attrs,
                     SeqStmt body, ffi::Array<Var> result_vars = {}, Span span = Span());

  explicit RegionStmt(ffi::ObjectPtr<RegionStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionStmt, Stmt, RegionStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RegionStmtNode);
};

/*!
 * \brief Assert condition, if an error occurs, return the error message.
 *
 * The error is described by:
 * - \p error_kind: the error kind (e.g. "RuntimeError", "TypeError", "ValueError")
 * - \p message_parts: an array of string fragments that are concatenated at runtime
 *   via TVMFFIErrorSetRaisedFromCStrParts. This enables string fragment reuse
 *   across multiple assertions to reduce binary size.
 */
class AssertStmtNode : public StmtNode {
 public:
  explicit AssertStmtNode(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}

  AssertStmtNode(PrimExpr condition, StringImm error_kind)
      : condition(std::move(condition)), error_kind(std::move(error_kind)) {}

  /*! \brief Condition to be checked. */
  PrimExpr condition;
  /*! \brief The error kind, e.g. "RuntimeError", "TypeError", "ValueError". */
  StringImm error_kind;
  /*! \brief Error message fragments, concatenated at runtime when assertion fails. */
  ffi::Array<StringImm> message_parts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertStmtNode>()
        .def_ro("condition", &AssertStmtNode::condition)
        .def_ro("error_kind", &AssertStmtNode::error_kind)
        .def_ro("message_parts", &AssertStmtNode::message_parts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.AssertStmt", AssertStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssertStmtNode.
 * \sa AssertStmtNode
 */
class AssertStmt : public Stmt {
 public:
  TVM_DLL AssertStmt(PrimExpr condition, StringImm error_kind, ffi::Array<StringImm> message_parts,
                     Span span = Span());

  explicit AssertStmt(ffi::ObjectPtr<AssertStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertStmt, Stmt, AssertStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtNode);
};

/*!
 * \brief Store value to the high dimension buffer.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 * \sa MakeTensorLoad
 */
class TensorStoreNode : public StmtNode {
 public:
  explicit TensorStoreNode(ffi::UnsafeInit tag) : buffer(tag), value(tag) {}

  TensorStoreNode(TensorVar buffer, PrimExpr value)
      : buffer(std::move(buffer)), value(std::move(value)) {}

  /*! \brief The buffer variable. */
  TensorVar buffer;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The indices location to be stored. */
  ffi::Array<PrimExpr> indices;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorStoreNode>()
        .def_ro("buffer", &TensorStoreNode::buffer)
        .def_ro("value", &TensorStoreNode::value)
        .def_ro("indices", &TensorStoreNode::indices);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TensorStore", TensorStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to TensorStoreNode.
 * \sa TensorStoreNode
 */
class TensorStore : public Stmt {
 public:
  TVM_DLL explicit TensorStore(TensorVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                               Span span = Span());

  explicit TensorStore(ffi::ObjectPtr<TensorStoreNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorStore, Stmt, TensorStoreNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TensorStoreNode);
};

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class EvaluateNode : public StmtNode {
 public:
  explicit EvaluateNode(ffi::UnsafeInit tag) : value(tag) {}

  explicit EvaluateNode(Expr value) : value(std::move(value)) {}

  /*! \brief The expression to be evaluated. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<EvaluateNode>().def_ro("value", &EvaluateNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Evaluate", EvaluateNode, StmtNode);
};

/*!
 * \brief Managed reference to EvaluateNode.
 * \sa EvaluateNode
 */
class Evaluate : public Stmt {
 public:
  TVM_DLL explicit Evaluate(Expr value, Span span = Span());

  explicit Evaluate(int value, Span span = Span()) : Evaluate(PrimExpr(value), span) {}

  explicit Evaluate(ffi::ObjectPtr<EvaluateNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Evaluate, Stmt, EvaluateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateNode);
};

/*!
 * \brief IfThenElse statement.
 */
class IfThenElseNode : public StmtNode {
 public:
  explicit IfThenElseNode(ffi::UnsafeInit tag) : condition(tag), then_case(tag) {}

  IfThenElseNode(PrimExpr condition, SeqStmt then_case)
      : condition(std::move(condition)), then_case(std::move(then_case)) {}

  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  SeqStmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  ffi::Optional<SeqStmt> else_case;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfThenElseNode>()
        .def_ro("condition", &IfThenElseNode::condition)
        .def_ro("then_case", &IfThenElseNode::then_case)
        .def_ro("else_case", &IfThenElseNode::else_case);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.IfThenElse", IfThenElseNode, StmtNode);
};

/*!
 * \brief Managed reference to IfThenElseNode.
 * \sa IfThenElseNode
 */
class IfThenElse : public Stmt {
 public:
  TVM_DLL IfThenElse(PrimExpr condition, SeqStmt then_case,
                     ffi::Optional<SeqStmt> else_case = std::nullopt, Span span = Span());

  explicit IfThenElse(ffi::ObjectPtr<IfThenElseNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfThenElse, Stmt, IfThenElseNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfThenElseNode);
};

/*!
 * \brief The kind of the loop.
 *
 *  ForKind can change the control flow semantics
 *  of the loop. So the kind field needs to be considered
 *  in all TIR passes.
 */
enum class ForKind : int {
  /*! \brief Ordinary loop with sequential iteration semantics. */
  kDefault = 0,
  /*! \brief Parallel execution, optionally placed by the thread_binding annotation. */
  kParallel = 1,
  /*!
   * \brief Vector SIMD loop.
   *  The loop body will be vectorized.
   */
  kVectorized = 2,
  /*! \brief The loop body must be unrolled. */
  kUnrolled = 3,
};

/*!
 * \brief A for loop, with possible type annotations.
 *
 * \code
 *
 *  for (loop_var = min; loop_var < min + extent; loop_var += step) {
 *    // body
 *  }
 * \endcode
 */
class ForNode : public StmtNode {
 public:
  explicit ForNode(ffi::UnsafeInit tag) : loop_var(tag), min(tag), extent(tag), body(tag) {}

  ForNode(PrimVar loop_var, PrimExpr min, PrimExpr extent, SeqStmt body)
      : loop_var(std::move(loop_var)),
        min(std::move(min)),
        extent(std::move(extent)),
        body(std::move(body)) {}

  /*! \brief The loop variable. */
  PrimVar loop_var;
  /*! \brief The minimum value of iteration. */
  PrimExpr min;
  /*! \brief The extent of the iteration. */
  PrimExpr extent;
  /*! \brief The kind of the for loop. */
  ForKind kind;
  /*! \brief The body of the for loop. */
  SeqStmt body;
  /*!
   * \brief Additional annotations about the loop.
   *
   *  The reserved thread_binding entry is a string specifying the thread tag
   *  for a parallel loop. It determines execution placement and must be
   *  preserved by transformations. Other entries provide auxiliary hints.
   */
  ffi::Map<ffi::String, ffi::Any> annotations;
  /*!
   * \brief The loop step. It is one if not specified.
   */
  ffi::Optional<PrimExpr> step;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForNode>()
        .def_ro("loop_var", &ForNode::loop_var, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("min", &ForNode::min)
        .def_ro("extent", &ForNode::extent)
        .def_ro("kind", &ForNode::kind)
        .def_ro("body", &ForNode::body)
        .def_ro("annotations", &ForNode::annotations)
        .def_ro("step", &ForNode::step);
  }

  /*! \brief Return the explicit thread placement, if present. */
  ffi::Optional<ffi::String> GetThreadBinding() const {
    if (auto value = annotations.Get("thread_binding")) {
      return value->cast<ffi::String>();
    }
    return std::nullopt;
  }

  /*! \brief Check it is a loop without nontrivial loop step. */
  bool HasTrivialStep() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.For", ForNode, StmtNode);
};

/*!
 * \brief Managed reference to ForNode.
 * \sa ForNode
 */
class For : public Stmt {
 public:
  TVM_DLL For(PrimVar loop_var, PrimExpr min, PrimExpr extent, ForKind kind, SeqStmt body,
              ffi::Map<ffi::String, ffi::Any> annotations = {},
              ffi::Optional<PrimExpr> step = std::nullopt, Span span = Span());

  explicit For(ffi::ObjectPtr<ForNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(For, Stmt, ForNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ForNode);
};

/*!
 * \brief A While loop
 *
 * \code
 *
 *  while (condition)
 *    body
 *
 * \endcode
 */
class WhileNode : public StmtNode {
 public:
  explicit WhileNode(ffi::UnsafeInit tag) : condition(tag), body(tag) {}

  WhileNode(PrimExpr condition, SeqStmt body)
      : condition(std::move(condition)), body(std::move(body)) {}

  /*! \brief The termination condition. */
  PrimExpr condition;
  /*! \brief The body of the while loop. */
  SeqStmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileNode>()
        .def_ro("condition", &WhileNode::condition)
        .def_ro("body", &WhileNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.While", WhileNode, StmtNode);
};

/*!
 * \brief Managed reference to WhileNode.
 * \sa WhileNode
 */
class While : public Stmt {
 public:
  TVM_DLL While(PrimExpr condition, SeqStmt body, Span span = Span());

  explicit While(ffi::ObjectPtr<WhileNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(While, Stmt, WhileNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileNode);
};

/*!
 * \brief A return from the current function.
 */
class ReturnNode : public StmtNode {
 public:
  explicit ReturnNode(ffi::UnsafeInit tag) : value(tag) {}

  explicit ReturnNode(Expr value) : value(std::move(value)) {}

  /*! \brief The value to return. */
  Expr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReturnNode>().def_ro("value", &ReturnNode::value);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Return", ReturnNode, StmtNode);
};

/*!
 * \brief Managed reference to ReturnNode.
 * \sa ReturnNode
 */
class Return : public Stmt {
 public:
  TVM_DLL explicit Return(Expr value, Span span = Span());

  explicit Return(ffi::ObjectPtr<ReturnNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Return, Stmt, ReturnNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ReturnNode);
};

/*!
 * \brief A Break in control flow.
 */
class BreakNode : public StmtNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BreakNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Break", BreakNode, StmtNode);
};

/*!
 * \brief Managed reference to BreakNode.
 * \sa BreakNode
 */
class Break : public Stmt {
 public:
  TVM_DLL explicit Break(Span span);

  explicit Break(ffi::ObjectPtr<BreakNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Break, Stmt, BreakNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BreakNode);
};

/*!
 * \brief A Continue in control flow.
 */
class ContinueNode : public StmtNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ContinueNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Continue", ContinueNode, StmtNode);
};

/*!
 * \brief Managed reference to ContinueNode.
 * \sa ContinueNode
 */
class Continue : public Stmt {
 public:
  TVM_DLL explicit Continue(Span span);

  explicit Continue(ffi::ObjectPtr<ContinueNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Continue, Stmt, ContinueNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ContinueNode);
};

/*!
 * \brief Standalone statement that declares a scope-id binding (e.g. cta_id,
 * warp_id, lane_id). Carries a ``ScopeIdDef`` value.
 *
 * Each declaration is a flat stmt within the device-region body. The declared
 * ``Var``\ s are visible in subsequent stmts in the same enclosing scope
 * (the ``tirx.device_entry`` region body), analogous to ``BindNode``.
 */
class ScopeIdDefStmtNode : public StmtNode {
 public:
  /*! \brief The scope-id definition (Vars + extents + binding). */
  ScopeIdDef def;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopeIdDefStmtNode>().def_ro("def", &ScopeIdDefStmtNode::def);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ScopeIdDefStmt", ScopeIdDefStmtNode, StmtNode);
};

/*! \brief Managed reference to ScopeIdDefStmtNode. */
class ScopeIdDefStmt : public Stmt {
 public:
  TVM_DLL ScopeIdDefStmt(ScopeIdDef def, Span span = Span());

  explicit ScopeIdDefStmt(ffi::ObjectPtr<ScopeIdDefStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeIdDefStmt, Stmt, ScopeIdDefStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdDefStmtNode);
};

/*! \brief Statement attribute and loop annotation keys. */
namespace attr {
/*!
 * \brief For annotation: maximum work for automatic unrolling.
 *
 * Integer policy inherited by nested loops unless they override it. Consumed by UnrollLoop.
 */
constexpr const char* auto_unroll_max_step = "auto_unroll_max_step";
/*!
 * \brief For annotation: expand unrolled bodies instead of preserving unrolled loops.
 *
 * Integer policy inherited by nested loops unless they override it. Consumed by UnrollLoop.
 */
constexpr const char* unroll_explicit = "unroll_explicit";
/*! \brief Annotation key on AllocTensor marking the allocation as volatile. */
constexpr const char* kVolatile = "tirx.volatile";
/*! \brief Mark buffer initial addr alignment in bytes */
constexpr const char* buffer_data_alignment = "buffer_data_alignment";
/*! \brief Mark buffer allocated addr in bytes */
constexpr const char* buffer_allocated_addr = "buffer_allocated_addr";

/*!
 * \brief Mark the kernel as persistent.
 */
constexpr const char* kPersistentKernel = "tirx.persistent_kernel";

}  // namespace attr
/*!
 * \brief Create a type annotation expression
 * \param dtype The data type
 * \param span The location of this object in the source code.
 * \return Expr a expression with dtype.
 */
TVM_DLL PrimExpr TypeAnnotation(PrimType dtype, Span span = Span());

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, ForKind kind);

// inline implementations
inline const char* ForKind2String(ForKind t) {
  switch (t) {
    case ForKind::kDefault:
      return "serial";
    case ForKind::kParallel:
      return "parallel";
    case ForKind::kVectorized:
      return "vectorized";
    case ForKind::kUnrolled:
      return "unroll";
  }
  TVM_FFI_THROW(InternalError) << "Unknown ForKind" << t;
  TVM_FFI_UNREACHABLE();
}

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_STMT_H_
