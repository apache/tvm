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
/*! \file tvm/ir/stmt.h
 * \brief Statements shared across IR dialects.
 */
#ifndef TVM_IR_STMT_H_
#define TVM_IR_STMT_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>

namespace tvm {

/*! \brief Base node of all statements. */
class StmtNode : public ffi::Object {
 public:
  /*!
   * \brief Location that points to the original source code.
   *        Reserved debug information.
   */
  mutable Location loc;

  StmtNode() = default;
  explicit StmtNode(Location loc) : loc(loc) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<StmtNode>().def_ro("loc", &StmtNode::loc,
                                       refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_s_eq_hash_subclass_kind_fixed = true;

  static constexpr const uint32_t _type_child_slots = 15;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Stmt", StmtNode, ffi::Object);
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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.SeqStmt", SeqStmtNode, StmtNode);
};

/*! \brief Sequence statement. */
class SeqStmt : public Stmt {
 public:
  /*!
   * \brief Construct SeqStmt.
   * \param seq The sequence.
   * \param loc The location of this object in the source code.
   */
  TVM_DLL explicit SeqStmt(ffi::Array<Stmt> seq, ffi::Optional<Location> loc = std::nullopt);
  /*! \brief Wrap a statement, reusing an existing sequence when possible. */
  TVM_DLL SeqStmt(Stmt stmt, ffi::Optional<Location> loc = std::nullopt);
  SeqStmt(std::initializer_list<Stmt> seq, ffi::Optional<Location> loc = std::nullopt)
      : SeqStmt(ffi::Array<Stmt>(seq), std::move(loc)) {}

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

namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<SeqStmt> = false;

template <>
struct TypeTraits<SeqStmt> : public ObjectRefWithFallbackTraitsBase<SeqStmt, Stmt, Array<Stmt>> {
  static SeqStmt ConvertFallbackValue(Stmt stmt) { return SeqStmt(std::move(stmt)); }
  static SeqStmt ConvertFallbackValue(Array<Stmt> seq) { return SeqStmt(std::move(seq)); }
};
}  // namespace ffi

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Bind", BindNode, StmtNode);
};

/*!
 * \brief Managed reference to BindNode.
 * \sa BindNode
 */
class Bind : public Stmt {
 public:
  TVM_DLL Bind(Var var, Expr value, ffi::Optional<Location> loc = std::nullopt);

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.RegionStmt", RegionStmtNode, StmtNode);
};

/*! \brief Managed reference to RegionStmtNode. */
class RegionStmt : public Stmt {
 public:
  TVM_DLL RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params, DictAttrs attrs,
                     SeqStmt body, ffi::Array<Var> result_vars = {},
                     ffi::Optional<Location> loc = std::nullopt);

  explicit RegionStmt(ffi::ObjectPtr<RegionStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionStmt, Stmt, RegionStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RegionStmtNode);
};

/*!
 * \brief Construct fresh typed variables for a region's lexical body parameters.
 *
 * The input carries only the operation, operands and attributes, without a body
 * or builder state. Parameters are ordered, distinct definitions and may have
 * name hints. Every region operation must register this hook, returning an empty
 * array when it has no body parameters. Presence of the attribute identifies
 * region support without invoking the hook or allocating variables.
 */
using FRegionGetBodyParams =
    ffi::reflection::NativeFunctionView<ffi::Array<Var>(const CallNode* call)>;

/*! \brief Optional dialect validation of a complete region statement. */
using FRegionValidate = ffi::reflection::NativeFunctionView<void(const RegionStmtNode*)>;

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.AssertStmt", AssertStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AssertStmtNode.
 * \sa AssertStmtNode
 */
class AssertStmt : public Stmt {
 public:
  TVM_DLL AssertStmt(PrimExpr condition, StringImm error_kind, ffi::Array<StringImm> message_parts,
                     ffi::Optional<Location> loc = std::nullopt);

  explicit AssertStmt(ffi::ObjectPtr<AssertStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertStmt, Stmt, AssertStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AssertStmtNode);
};

/*!
 * \brief Store a primitive value to an indexed destination expression.
 *
 * \code
 *
 *  buffer[i, j] = value;
 *
 * \endcode
 */
class TensorStoreNode : public StmtNode {
 public:
  explicit TensorStoreNode(ffi::UnsafeInit tag) : dest(tag), value(tag) {}

  TensorStoreNode(Expr dest, PrimExpr value) : dest(std::move(dest)), value(std::move(value)) {}

  /*! \brief The indexed destination expression. */
  Expr dest;
  /*! \brief The indices location to be stored. */
  ffi::Array<PrimExpr> indices;
  /*! \brief The value to be stored. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorStoreNode>()
        .def_ro("dest", &TensorStoreNode::dest)
        .def_ro("indices", &TensorStoreNode::indices)
        .def_ro("value", &TensorStoreNode::value);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.TensorStore", TensorStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to TensorStoreNode.
 * \sa TensorStoreNode
 */
class TensorStore : public Stmt {
 public:
  TVM_DLL explicit TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value,
                               ffi::Optional<Location> loc = std::nullopt);

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Evaluate", EvaluateNode, StmtNode);
};

/*!
 * \brief Managed reference to EvaluateNode.
 * \sa EvaluateNode
 */
class Evaluate : public Stmt {
 public:
  TVM_DLL explicit Evaluate(Expr value, ffi::Optional<Location> loc = std::nullopt);

  explicit Evaluate(int value, ffi::Optional<Location> loc = std::nullopt)
      : Evaluate(PrimExpr(value), loc) {}

  explicit Evaluate(ffi::ObjectPtr<EvaluateNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Evaluate, Stmt, EvaluateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateNode);
};

/*!
 * \brief If statement.
 */
class IfNode : public StmtNode {
 public:
  explicit IfNode(ffi::UnsafeInit tag) : condition(tag), then_case(tag) {}

  IfNode(PrimExpr condition, SeqStmt then_case)
      : condition(std::move(condition)), then_case(std::move(then_case)) {}

  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  SeqStmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  ffi::Optional<SeqStmt> else_case;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfNode>()
        .def_ro("condition", &IfNode::condition)
        .def_ro("then_case", &IfNode::then_case)
        .def_ro("else_case", &IfNode::else_case);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.If", IfNode, StmtNode);
};

/*!
 * \brief Managed reference to IfNode.
 * \sa IfNode
 */
class If : public Stmt {
 public:
  TVM_DLL If(PrimExpr condition, SeqStmt then_case, ffi::Optional<SeqStmt> else_case = std::nullopt,
             ffi::Optional<Location> loc = std::nullopt);

  explicit If(ffi::ObjectPtr<IfNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(If, Stmt, IfNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfNode);
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
  /*! \brief Parallel execution. */
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
   * Entries are interpreted by the consuming dialect. Transformations preserve
   * annotations unless they explicitly handle the corresponding semantics.
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

  /*! \brief Check it is a loop without nontrivial loop step. */
  bool HasTrivialStep() const;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.For", ForNode, StmtNode);
};

/*!
 * \brief Managed reference to ForNode.
 * \sa ForNode
 */
class For : public Stmt {
 public:
  TVM_DLL For(PrimVar loop_var, PrimExpr min, PrimExpr extent, ForKind kind, SeqStmt body,
              ffi::Map<ffi::String, ffi::Any> annotations = {},
              ffi::Optional<PrimExpr> step = std::nullopt,
              ffi::Optional<Location> loc = std::nullopt);

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.While", WhileNode, StmtNode);
};

/*!
 * \brief Managed reference to WhileNode.
 * \sa WhileNode
 */
class While : public Stmt {
 public:
  TVM_DLL While(PrimExpr condition, SeqStmt body, ffi::Optional<Location> loc = std::nullopt);

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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Return", ReturnNode, StmtNode);
};

/*!
 * \brief Managed reference to ReturnNode.
 * \sa ReturnNode
 */
class Return : public Stmt {
 public:
  TVM_DLL explicit Return(Expr value, ffi::Optional<Location> loc = std::nullopt);

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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Break", BreakNode, StmtNode);
};

/*!
 * \brief Managed reference to BreakNode.
 * \sa BreakNode
 */
class Break : public Stmt {
 public:
  TVM_DLL explicit Break(ffi::Optional<Location> loc);

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

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Continue", ContinueNode, StmtNode);
};

/*!
 * \brief Managed reference to ContinueNode.
 * \sa ContinueNode
 */
class Continue : public Stmt {
 public:
  TVM_DLL explicit Continue(ffi::Optional<Location> loc);

  explicit Continue(ffi::ObjectPtr<ContinueNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Continue, Stmt, ContinueNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ContinueNode);
};

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

namespace op_attr {
inline constexpr const char* kRegionGetBodyParams = "FRegionGetBodyParams";
inline constexpr const char* kRegionValidate = "FRegionValidate";
}  // namespace op_attr

namespace type_attr {
inline constexpr const char* kEvaluateValidate = "__evaluate_validate__";
inline constexpr const char* kTensorStoreValidate = "__tensor_store_validate__";
}  // namespace type_attr

}  // namespace tvm
#endif  // TVM_IR_STMT_H_
