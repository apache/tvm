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
#ifndef TVM_TIR_STMT_H_
#define TVM_TIR_STMT_H_

#include <tvm/ffi/ir/traits.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/script_printer.h>
#include <tvm/tirx/expr.h>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace tirx {

/*! \brief Base node of all statements. */
class StmtNode : public Object {
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
    refl::ObjectDef<StmtNode>().def_ro("span", &StmtNode::span);
  }

  TVM_OBJECT_ENABLE_SCRIPT_PRINTER();

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  static constexpr const uint32_t _type_child_slots = 15;
  TVM_FFI_DECLARE_OBJECT_INFO("tirx.Stmt", StmtNode, Object);
};

/*! \brief Container of all statements */
class Stmt : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Stmt, ObjectRef, StmtNode);
};

/*!
 * \brief Bind a variable to a value in the enclosing scope.
 *
 * BindNode has no body field. The bound variable is visible
 * in all subsequent statements within the same enclosing scope (SeqStmt,
 * ForNode.body, etc.). This enables flat (non-nested) IR sequences.
 */
class BindNode : public StmtNode {
 public:
  /*! \brief The variable being bound. */
  Var var;
  /*! \brief The value to bind to the variable. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<BindNode>()
        .def_ro("var", &BindNode::var, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("value", &BindNode::value)
        .def_ir_traits<tr::AssignObj>("$field:var", "$field:value");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Bind", BindNode, StmtNode);
};

/*!
 * \brief Managed reference to BindNode.
 * \sa BindNode
 */
class Bind : public Stmt {
 public:
  TVM_DLL Bind(Var var, PrimExpr value, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Bind, Stmt, BindNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BindNode);
};

/*!
 * \brief Define certain auxiliary attribute for the body to be a symbolic value.
 *  This provide auxiliary information for IR passes that transforms body.
 *
 *  In terms of effect, this is equivalent to Block(Evaluate(value), body).
 *
 *  Examples of possible usage:
 *    - Bound of function, variables.
 *    - Hint which block corresponds to a parallel region.
 */
class AttrStmtNode : public StmtNode {
 public:
  /*! \brief this is attribute about certain node */
  ffi::Any node;
  /*! \brief the type key of the attribute */
  ffi::String attr_key;
  /*! \brief The attribute value, value is well defined at current scope. */
  PrimExpr value;
  /*! \brief The body statement to be executed */
  Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttrStmtNode>()
        .def_ro("node", &AttrStmtNode::node)
        .def_ro("attr_key", &AttrStmtNode::attr_key)
        .def_ro("value", &AttrStmtNode::value)
        .def_ro("body", &AttrStmtNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.AttrStmt", AttrStmtNode, StmtNode);
};

/*!
 * \brief Managed reference to AttrStmtNode.
 * \sa AttrStmtNode
 */
class AttrStmt : public Stmt {
 public:
  TVM_DLL AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, Stmt body,
                   Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AttrStmt, Stmt, AttrStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AttrStmtNode);
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
  /*! \brief Condition to be checked. */
  PrimExpr condition;
  /*! \brief The error kind, e.g. "RuntimeError", "TypeError", "ValueError". */
  StringImm error_kind;
  /*! \brief Error message fragments, concatenated at runtime when assertion fails. */
  ffi::Array<StringImm> message_parts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<AssertStmtNode>()
        .def_ro("condition", &AssertStmtNode::condition)
        .def_ro("error_kind", &AssertStmtNode::error_kind)
        .def_ro("message_parts", &AssertStmtNode::message_parts)
        .def_ir_traits<tr::AssertObj>("$field:condition", "$global:tirx._structured_msg");
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

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AssertStmt, Stmt, AssertStmtNode);
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
 * \sa BufferLoad
 */
class BufferStoreNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Buffer buffer;
  /*! \brief The value to be stored. */
  PrimExpr value;
  /*! \brief The indices location to be stored. */
  ffi::Array<PrimExpr> indices;
  /*! \brief The predicate mask for storing values. */
  ffi::Optional<PrimExpr> predicate;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<BufferStoreNode>()
        .def_ro("buffer", &BufferStoreNode::buffer)
        .def_ro("value", &BufferStoreNode::value)
        .def_ro("indices", &BufferStoreNode::indices)
        .def_ro("predicate", &BufferStoreNode::predicate)
        .def_ir_traits<tr::StoreObj>("$field:buffer", "$field:value",
                                    "$global:tirx._store_indices", "$field:predicate");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferStore", BufferStoreNode, StmtNode);
};

/*!
 * \brief Managed reference to BufferStoreNode.
 * \sa BufferStoreNode
 */
class BufferStore : public Stmt {
 public:
  TVM_DLL explicit BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                               ffi::Optional<PrimExpr> predicate = std::nullopt,
                               Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferStore, Stmt, BufferStoreNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferStoreNode);
};

/*! \brief Declare a buffer that can be used in the body */
class DeclBufferNode : public StmtNode {
 public:
  /*! \brief The buffer being declared */
  Buffer buffer;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DeclBufferNode>()
        .def_ro("buffer", &DeclBufferNode::buffer);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.DeclBuffer", DeclBufferNode, StmtNode);
};

/*! \brief Managed reference to DeclBufferNode */
class DeclBuffer : public Stmt {
 public:
  TVM_DLL DeclBuffer(Buffer buffer, Span span = Span());
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DeclBuffer, Stmt, DeclBufferNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DeclBufferNode);
};

/*! \brief Allocate a buffer and declare it in scope */
class AllocBufferNode : public StmtNode {
 public:
  /*! \brief The buffer being allocated and declared */
  Buffer buffer;
  /*!
   * \brief Additional annotations about the allocation.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations.
   */
  ffi::Map<ffi::String, ffi::Any> annotations;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllocBufferNode>()
        .def_ro("buffer", &AllocBufferNode::buffer, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("annotations", &AllocBufferNode::annotations);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.AllocBuffer", AllocBufferNode, StmtNode);
};

/*! \brief Managed reference to AllocBufferNode */
class AllocBuffer : public Stmt {
 public:
  TVM_DLL AllocBuffer(
      Buffer buffer,
      ffi::Map<ffi::String, ffi::Any> annotations = ffi::Map<ffi::String, ffi::Any>(),
      Span span = Span());
  /*!
   * \brief If the buffer's shape is constant, return the total number of elements.
   * \return The product of all shape extents if all are constant, std::nullopt otherwise.
   */
  std::optional<int64_t> ConstantAllocationSize() const {
    int64_t result = 1;
    for (const PrimExpr& extent : (*this)->buffer->shape) {
      if (const auto* int_size = extent.as<IntImmNode>()) {
        result *= int_size->value;
      } else {
        return std::nullopt;
      }
    }
    return result;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AllocBuffer, Stmt, AllocBufferNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AllocBufferNode);
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

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<SeqStmtNode>()
        .def_ro("seq", &SeqStmtNode::seq)
        .def_ir_traits<tr::SeqObj>("$field:seq");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.SeqStmt", SeqStmtNode, StmtNode);
};

/*!
 * \brief Evaluates an expression.
 *  This is mostly used for putting a Call node into Stmt.
 *
 *  If value do not have side-effect, this node can be safely removed.
 */
class EvaluateNode : public StmtNode {
 public:
  /*! \brief The expression to be evaluated. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<EvaluateNode>()
        .def_ro("value", &EvaluateNode::value)
        .def_ir_traits<tr::ExprStmtObj>("$global:tirx._evaluate_expr",
                                         "$global:tirx._evaluate_kind",
                                         "$global:tirx._evaluate_is_return");
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.Evaluate", EvaluateNode, StmtNode);
};

/*!
 * \brief Managed reference to EvaluateNode.
 * \sa EvaluateNode
 */
class Evaluate : public Stmt {
 public:
  TVM_DLL explicit Evaluate(PrimExpr value, Span span = Span());

  explicit Evaluate(int value, Span span = Span()) : Evaluate(PrimExpr(value), span) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Evaluate, Stmt, EvaluateNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(EvaluateNode);
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

  /*! \return get the size of the sequence */
  size_t size() const { return operator->()->size(); }
  /*!
   * \brief Get the index-th element in the sequence.
   */
  Stmt operator[](size_t index) const { return (*(operator->()))[index]; }
  /*!
   * \brief Construct a sequence statement by flattening
   *        all the arrays and sequences in the arguments
   *        recursively.
   *
   * - When an argument is nullptr, it will be ignored.
   * - When an argument is an array or a SeqStmt, it will be flattened recursively.
   * - A normal Stmt will be appended to the end of the sequence.
   *
   * \note This function can directly return an element
   *       if it is the only element in the sequence.
   *
   * \note If the only argument to this function is a SeqStmt, and if
   *       no flattening of the SeqStmt is required, then the SeqStmt
   *       will be returned as-is.
   *
   * \param seq_args The list of arguments to be flattened.
   * \tparam Args arguments
   * \return The constructed statement
   */
  template <typename... Args>
  static Stmt Flatten(Args&&... seq_args) {
    ffi::Array<Stmt> seq;

    ffi::details::for_each(Flattener(&seq), std::forward<Args>(seq_args)...);

    if (seq.empty()) {
      return Evaluate(0);
    } else if (seq.size() == 1) {
      return seq[0];
    }

    // If the argument is a single SeqStmt argument with no
    // flattening or unwrapping required, then we may
    // return the SeqStmt as-is.
    if constexpr (sizeof...(seq_args) == 1) {
      if (auto opt = Flattener::AsSeqStmt(std::forward<Args>(seq_args)...)) {
        SeqStmt original = opt.value();
        bool all_same = [&]() {
          if (original->seq.size() != seq.size()) {
            return false;
          }
          for (size_t i = 0; i < seq.size(); i++) {
            if (!original->seq[i].same_as(seq[i])) {
              return false;
            }
          }
          return true;
        }();
        if (all_same) {
          return original;
        }
      }
    }

    return SeqStmt(seq);
  }
  /*! \brief Helper class to flatten sequence of arguments into Array. */
  class Flattener {
   public:
    explicit Flattener(ffi::Array<Stmt>* seq) : seq_(seq) {}

    template <typename T>
    static ffi::Optional<SeqStmt> AsSeqStmt(const T& t) {
      if constexpr (std::is_same_v<T, SeqStmt>) {
        return t;
      }
      if constexpr (!std::is_base_of_v<T, SeqStmt>) {
        return std::nullopt;
      }
      if constexpr (std::is_base_of_v<Stmt, T>) {
        if (const SeqStmtNode* ptr = t.template as<SeqStmtNode>()) {
          return ffi::GetRef<SeqStmt>(ptr);
        } else {
          return std::nullopt;
        }
      }
      return std::nullopt;
    }

    template <typename T>
    void operator()(size_t i, const T& stmt_or_seq) const {
      if constexpr (std::is_base_of_v<ObjectRef, T>) {
        // Early bail-out, applicable to any ObjectRef
        if (!stmt_or_seq.defined()) {
          return;
        }
      }

      if constexpr (std::is_same_v<T, SeqStmt>) {
        // Static type-checking for a SeqStmt that could be flattened.
        (*this)(0, stmt_or_seq->seq);
        return;
      }

      if constexpr (std::is_base_of_v<T, SeqStmt>) {
        // Dynamic type-checking for a SeqStmt that could be
        // flattened.
        if (auto* op = stmt_or_seq.template as<SeqStmtNode>()) {
          operator()(0, op->seq);
          return;
        }
      }

      if constexpr (std::is_base_of_v<T, Evaluate>) {
        // Evaluate(0) is used to represent a no-op, and may be
        // generated by previous calls to SeqStmt::Flatten().  These
        // should be removed to ensure that Flatten(a+b) is equivalent
        // to Flatten(Flatten(a), Flatten(b)).
        if (auto* op = stmt_or_seq.template as<EvaluateNode>()) {
          if (auto* as_int = op->value.template as<IntImmNode>(); as_int && as_int->value == 0) {
            return;
          }
        }
      }

      if constexpr (std::is_base_of_v<Stmt, T>) {
        // Any other Stmt type just gets appended.
        seq_->push_back(stmt_or_seq);
      } else {
        // Anything else is treated as an iterable of Stmt.
        for (auto v : stmt_or_seq) {
          this->operator()(0, v);
        }
      }
    }

   private:
    ffi::Array<Stmt>* seq_;
  };

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SeqStmt, Stmt, SeqStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqStmtNode);
};

/*!
 * \brief IfThenElse statement.
 */
class IfThenElseNode : public StmtNode {
 public:
  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief The branch to be executed when condition is true. */
  Stmt then_case;
  /*! \brief The branch to be executed when condition is false, can be null. */
  ffi::Optional<Stmt> else_case;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<IfThenElseNode>()
        .def_ro("condition", &IfThenElseNode::condition)
        .def_ro("then_case", &IfThenElseNode::then_case)
        .def_ro("else_case", &IfThenElseNode::else_case)
        .def_ir_traits<tr::IfObj>("$field:condition",
                                  tr::Region("$field:then_case"),
                                  ffi::Optional<tr::Region>(tr::Region("$field:else_case")));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.IfThenElse", IfThenElseNode, StmtNode);
};

/*!
 * \brief Managed reference to IfThenElseNode.
 * \sa IfThenElseNode
 */
class IfThenElse : public Stmt {
 public:
  TVM_DLL IfThenElse(PrimExpr condition, Stmt then_case,
                     ffi::Optional<Stmt> else_case = std::nullopt, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IfThenElse, Stmt, IfThenElseNode);
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
  /*! \brief default semantics -- serial execution. */
  kSerial = 0,
  /*! \brief Parallel execution on CPU. */
  kParallel = 1,
  /*!
   * \brief Vector SIMD loop.
   *  The loop body will be vectorized.
   */
  kVectorized = 2,
  /*! \brief The loop body must be unrolled. */
  kUnrolled = 3,
  /*!
   * \brief The loop variable is bound to a thread in
   * an environment. In the final stage of lowering,
   * the loop is simply removed and the loop variable is
   * mapped to the corresponding context thread.
   */
  kThreadBinding = 4
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
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  PrimExpr min;
  /*! \brief The extent of the iteration. */
  PrimExpr extent;
  /*! \brief The kind of the for loop. */
  ForKind kind;
  /*! \brief The body of the for loop. */
  Stmt body;
  /*!
   * \brief Only valid when kind == ForKind::kThreadBinding
   * The context thread that this loop variable bounds to.
   */
  ffi::Optional<IterVar> thread_binding;
  /*!
   * \brief Additional annotations about the loop.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations. An annotation should
   *  not change the control flow semantics of the loop
   *  and can be ignored in most passes.
   */
  ffi::Map<ffi::String, ffi::Any> annotations;
  /*!
   * \brief The loop step. It is one if not specified.
   */
  ffi::Optional<PrimExpr> step;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForNode>()
        .def_ro("loop_var", &ForNode::loop_var, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("min", &ForNode::min)
        .def_ro("extent", &ForNode::extent)
        .def_ro("kind", &ForNode::kind)
        .def_ro("body", &ForNode::body)
        .def_ro("thread_binding", &ForNode::thread_binding)
        .def_ro("annotations", &ForNode::annotations)
        .def_ro("step", &ForNode::step);
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
  TVM_DLL For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
              ffi::Optional<IterVar> thread_binding = std::nullopt,
              ffi::Map<ffi::String, ffi::Any> annotations = {},
              ffi::Optional<PrimExpr> step = std::nullopt, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(For, Stmt, ForNode);
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
  /*! \brief The termination condition. */
  PrimExpr condition;
  /*! \brief The body of the while loop. */
  Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<WhileNode>()
        .def_ro("condition", &WhileNode::condition)
        .def_ro("body", &WhileNode::body)
        .def_ir_traits<tr::WhileObj>("$field:condition", tr::Region("$field:body"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.While", WhileNode, StmtNode);
};

/*!
 * \brief Managed reference to WhileNode.
 * \sa WhileNode
 */
class While : public Stmt {
 public:
  TVM_DLL While(PrimExpr condition, Stmt body, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(While, Stmt, WhileNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(WhileNode);
};

/*!
 * \brief Representing the region of multi-dimensional buffer access.
 */
class BufferRegionNode : public PrimExprConvertibleNode {
 public:
  /*! \brief The buffer of the buffer region. */
  Buffer buffer;
  /*! \brief The region array of the buffer region. */
  ffi::Array<Range> region;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<BufferRegionNode>()
        .def_ro("buffer", &BufferRegionNode::buffer)
        .def_ro("region", &BufferRegionNode::region)
        .def_ir_traits<tr::LoadObj>("$field:buffer", "$global:tirx._buf_region_indices");
  }

  TVM_DLL PrimExpr ToPrimExpr() const final;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.BufferRegion", BufferRegionNode, PrimExprConvertibleNode);
};

/*!
 * \brief Managed reference to BufferRegionNode.
 * \sa BufferRegionNode
 */
class BufferRegion : public PrimExprConvertible {
 public:
  TVM_DLL explicit BufferRegion(Buffer buffer, ffi::Array<Range> region);

  /*!
   * \brief Create a BufferRegion which is full region of the given buffer.
   * \param buffer The buffer to generate full BufferRegion.
   * \return The BufferRegion which covers all region of the given buffer
   */
  TVM_DLL static BufferRegion FullRegion(Buffer buffer);

  /*!
   * \brief Create a BufferRegion which is a single point of the given buffer.
   * \param buffer The buffer to generate single point BufferRegion.
   * \param indices The access point indices of the buffer
   * \return The BufferRegion which is the single point of the given buffer.
   */
  TVM_DLL static BufferRegion FromPoint(Buffer buffer, ffi::Array<PrimExpr> indices);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferRegion, PrimExprConvertible, BufferRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferRegionNode);
};

/*!
 * \brief Match introduces a constraint that the source buffer region can be remapped to the data
 * layout specified by the buffer field. The constraint can be checked in later part of lowering (or
 * optionally during runtime).
 *
 * MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
 * low-level hardware primitives in the IR and defer the check after the sequence of
 * transformations.
 */
class MatchBufferRegionNode : public Object {
 public:
  /*! \brief The target buffer. */
  Buffer buffer;
  /*! \brief The source buffer region. */
  BufferRegion source;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchBufferRegionNode>()
        .def_ro("buffer", &MatchBufferRegionNode::buffer)
        .def_ro("source", &MatchBufferRegionNode::source);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.MatchBufferRegion", MatchBufferRegionNode, Object);
};

/*!
 * \brief Managed reference to MatchBufferRegionNode.
 * \sa MatchBufferRegionNode
 */
class MatchBufferRegion : public ObjectRef {
 public:
  TVM_DLL explicit MatchBufferRegion(Buffer buffer, BufferRegion source);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MatchBufferRegion, ObjectRef, MatchBufferRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchBufferRegionNode);
};

/*!
 * \brief A block is a basic schedule unit in TIR.
 * \note SBlock's body is parameterized by iter vars.
 * \code
 *
 *  with T.sblock(name):
 *      v0 = T.axis.S(domain, value0)
 *      v1 = T.axis.R(domain, value1)
 *      ...
 *      T.reads([buffer0[start:end, ...], ...])
 *      T.writes([buffer1[start:end, ...], ...])
 *      T.where(predicate)
 *      buffer2 = T.alloc_buffer(shape, dtype)
 *      buffer3 = T.match_buffer(source_buffer[start:end, ...])
 *      T.attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class SBlockNode : public StmtNode {
 public:
  /*! \brief The variables of the block. */
  ffi::Array<IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  ffi::Array<BufferRegion> reads;
  /*! \brief The write buffer regions of the block. */
  ffi::Array<BufferRegion> writes;
  /*! \brief The name_hint of the block. */
  ffi::String name_hint;
  /*! \brief The buffer allocated in the block. */
  ffi::Array<Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  ffi::Array<MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  ffi::Map<ffi::String, ffi::Any> annotations;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `std::nullopt` if there is no reduction iter_vars
   */
  ffi::Optional<Stmt> init;
  /*! \brief The body of the block. */
  Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<SBlockNode>()
        .def_ro("iter_vars", &SBlockNode::iter_vars, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("reads", &SBlockNode::reads)
        .def_ro("writes", &SBlockNode::writes)
        .def_ro("name_hint", &SBlockNode::name_hint, refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("alloc_buffers", &SBlockNode::alloc_buffers)
        .def_ro("match_buffers", &SBlockNode::match_buffers)
        .def_ro("annotations", &SBlockNode::annotations)
        .def_ro("init", &SBlockNode::init)
        .def_ro("body", &SBlockNode::body)
        .def_ir_traits<tr::WithObj>(tr::Region("$field:body", "$field:iter_vars"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.SBlock", SBlockNode, StmtNode);
};

/*!
 * \brief Managed reference to SBlockNode.
 * \sa SBlockNode
 */
class SBlock : public Stmt {
 public:
  TVM_DLL explicit SBlock(
      ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
      ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
      ffi::Optional<Stmt> init = std::nullopt,
      ffi::Array<Buffer> alloc_buffers = ffi::Array<Buffer>(),
      ffi::Array<MatchBufferRegion> match_buffers = ffi::Array<MatchBufferRegion>(),
      ffi::Map<ffi::String, ffi::Any> annotations = ffi::Map<ffi::String, ffi::Any>(),
      Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SBlock, Stmt, SBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockNode);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class SBlockRealizeNode : public StmtNode {
 public:
  /*! \brief The corresponding values of the iter vars. */
  ffi::Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  PrimExpr predicate;
  /*! \brief The block to be realized. */
  SBlock block;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    namespace tr = tvm::ffi::ir::traits;
    refl::ObjectDef<SBlockRealizeNode>()
        .def_ro("iter_values", &SBlockRealizeNode::iter_values)
        .def_ro("predicate", &SBlockRealizeNode::predicate)
        .def_ro("block", &SBlockRealizeNode::block)
        .def_ir_traits<tr::WithObj>(tr::Region("$field:block"));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.SBlockRealize", SBlockRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to BlockRealizeNode
 * \sa BlockRealizeNode
 */
class SBlockRealize : public Stmt {
 public:
  TVM_DLL explicit SBlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, SBlock block,
                                 Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SBlockRealize, Stmt, SBlockRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockRealizeNode);
};

/*! \brief namespace of possible attributes in AttrStmt.attr_key */
namespace attr {
/*! \brief Mark stores/loads with their bounds. */
constexpr const char* buffer_bound = "buffer_bound";
/*!
 * \brief Mark the scope as when computation start to happen.
 *  This can hint some code generator to create a new function for compute.
 */
constexpr const char* compute_scope = "compute_scope";
/*! \brief The allocation device for global malloc in host. */
constexpr const char* device_id = "device_id";
/*! \brief Mark that it is in the device scope. */
constexpr const char* device_scope = "device_scope";
/*! \brief The device type. */
constexpr const char* device_type = "device_type";
/*!
 * \brief Mark the scope as generated by extern primitive.
 *  Such scope can contain arbitrary ir program and we need to be careful
 *  when making certain assumptions about the structure of the program.
 */
constexpr const char* extern_scope = "extern_scope";
/*! \brief Pragma: auto-unroll, max_step */
constexpr const char* pragma_auto_unroll_max_step = "pragma_auto_unroll_max_step";
/*! \brief Import C source or file into the final code gen module */
constexpr const char* pragma_import_c = "pragma_import_c";
/*! \brief Import llvm source or file into the final code gen module */
constexpr const char* pragma_import_llvm = "pragma_import_llvm";
/*! \brief Mark region is guarded by the pragma extension */
constexpr const char* pragma_scope_prefix = "pragma_";
/*! \brief Try to modify the AST to support Tensor Core */
constexpr const char* pragma_tensor_core = "pragma_tensor_core";
/*! \brief Pragma: unroll explicit */
constexpr const char* pragma_unroll_explicit = "pragma_unroll_explicit";
/*! \brief Mark storage alignment requirement of buffers */
constexpr const char* storage_alignment = "storage_alignment";
/*! \brief Mark launching extent of thread, used by device API. */
constexpr const char* thread_extent = "thread_extent";
/*! \brief Annotation key on AllocBuffer marking the allocation as volatile. */
constexpr const char* kVolatile = "tirx.volatile";

/*!
 * \brief Check if attr_key is a pragma key extension
 * \param attr_key The attr key to be compared
 * \return true if it is a pragma key
 */
inline bool IsPragmaKey(const std::string& attr_key) {
  return attr_key.compare(0, 7, "pragma_") == 0;
}

}  // namespace attr
/*!
 * \brief Create a type annotation expression
 * \param dtype The data type
 * \param span The location of this object in the source code.
 * \return Expr a expression with dtype.
 */
TVM_DLL PrimExpr TypeAnnotation(DataType dtype, Span span = Span());

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, ForKind kind);

// inline implementations
inline const char* ForKind2String(ForKind t) {
  switch (t) {
    case ForKind::kSerial:
      return "serial";
    case ForKind::kParallel:
      return "parallel";
    case ForKind::kVectorized:
      return "vectorized";
    case ForKind::kUnrolled:
      return "unroll";
    case ForKind::kThreadBinding:
      return "thread_binding";
  }
  TVM_FFI_THROW(InternalError) << "Unknown ForKind" << t;
  TVM_FFI_UNREACHABLE();
}

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_STMT_H_
