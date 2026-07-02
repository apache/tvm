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
 * \file tirx/ir/tir_visitor_with_path.h
 * \brief Provide a TIR visitor that tracks the current location
 */
#ifndef TVM_TIRX_IR_TIR_VISITOR_WITH_PATH_H_
#define TVM_TIRX_IR_TIR_VISITOR_WITH_PATH_H_

#include <tvm/ir/module.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/stmt_functor.h>

#include <exception>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace tvm {
namespace tirx {

/*! \brief Visit TIR while tracking the ffi::reflection::AccessPath */
class TIRVisitorWithPath
    : protected ExprFunctor<void(const PrimExpr&, ffi::reflection::AccessPath)>,
      protected StmtFunctor<void(const Stmt&, ffi::reflection::AccessPath)> {
 public:
  template <typename TObjectRef>
  void operator()(TObjectRef&& obj) {
    Visit(std::forward<TObjectRef>(obj), ffi::reflection::AccessPath::Root());
  }

 protected:
  // Delegate to ExprFunctor::VisitExpr for PrimExpr, and any subclasses
  virtual inline void Visit(const PrimExpr& obj, ffi::reflection::AccessPath path) {
    VisitExpr(obj, path);
  }
  // Core Call stores arguments as Expr, including pointer-typed Vars.
  virtual inline void Visit(const Expr& obj, ffi::reflection::AccessPath path) {
    if (auto prim = obj.as<PrimExpr>()) {
      Visit(prim.value(), path);
    } else if (auto* var = obj.as<VarNode>()) {
      VisitExpr_(var, path);
    } else if (auto* call = obj.as<CallNode>()) {
      VisitExpr_(call, path);
    } else {
      TVM_FFI_THROW(TypeError) << "Unsupported non-primitive TIR expression "
                               << obj.GetTypeKey();
    }
  }
  // Delegate to ExprFunctor::VisitStmt for Stmt, and any subclasses
  virtual inline void Visit(const Stmt& obj, ffi::reflection::AccessPath path) {
    VisitStmt(obj, path);
  }

  // Visit a buffer at a use site (BufferLoad, BufferStore, reads/writes).
  // By default, does not re-visit buffer fields (shape, strides, elem_offset),
  // as those are visited at the definition site via EnterDef.
  virtual void VisitBufferUse(const Buffer& obj, ffi::reflection::AccessPath path);
  // Visit a buffer at a definition site. By default visits buffer fields.
  virtual void VisitBufferDef(const Buffer& obj, ffi::reflection::AccessPath path);

  // Visitors for TIR constructs that are neither PrimExpr nor Stmt
  virtual void Visit(const IRModule& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const PrimFunc& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const GlobalVar& obj, ffi::reflection::AccessPath path) {}
  virtual void Visit(const Range& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const BufferRegion& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const MatchBufferRegion& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const IterVar& obj, ffi::reflection::AccessPath path);

  // Called when entering/exiting the scope of a GlobalVar definition.
  virtual void EnterDef(const GlobalVar& var, ffi::reflection::AccessPath path) {}
  virtual void ExitDef(const GlobalVar& var, ffi::reflection::AccessPath path) {}

  // Called when entering/exiting the scope of a tirx::Var definition.
  virtual void EnterDef(const Var& var, ffi::reflection::AccessPath path) {}
  virtual void ExitDef(const Var& var, ffi::reflection::AccessPath path) {}

  // Called when entering/exiting the scope of an IterVar definition.
  // By default, visits the `Range IterVarNode::dom`, then enters the
  // scope of the internal `tirx::Var`.
  virtual void EnterDef(const IterVar& var, ffi::reflection::AccessPath path);
  virtual void ExitDef(const IterVar& var, ffi::reflection::AccessPath path);

  // Called when entering/exiting the scope of a Buffer definition.
  // By default, visits the buffer's data pointer, shape, strides, and
  // elem_offset, which must be defined prior to defining the Buffer.
  virtual void EnterDef(const Buffer& buffer, ffi::reflection::AccessPath path);
  virtual void ExitDef(const Buffer& buffer, ffi::reflection::AccessPath path);

  // Utility to visit an array of nodes
  template <typename T>
  inline void Visit(const ffi::Array<T>& arr, ffi::reflection::AccessPath path) {
    for (size_t i = 0; i < arr.size(); i++) {
      Visit(arr[i], path->ArrayItem(i));
    }
  }

  // Utility to visit an optional node nodes
  template <typename T>
  inline void Visit(const ffi::Optional<T>& opt, ffi::reflection::AccessPath path) {
    if (opt) {
      Visit(opt.value(), path);
    }
  }

  using StmtFunctor::VisitStmt;
  void VisitStmt_(const BindNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AttrStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const IfThenElseNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const ForNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const WhileNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BreakNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const ContinueNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AllocBufferNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const DeclBufferNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BufferStoreNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AssertStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const SeqStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const EvaluateNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const SBlockNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const SBlockRealizeNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const tirx::TilePrimitiveCallNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const ScopeIdDefStmtNode* op, ffi::reflection::AccessPath path) override;

  using ExprFunctor::VisitExpr;
  void VisitExpr_(const VarNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const SizeVarNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const BufferLoadNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const ProducerLoadNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const LetNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const CallNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const AddNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const SubNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const MulNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const DivNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const ModNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const FloorDivNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const FloorModNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const MinNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const MaxNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const EQNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const NENode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const LTNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const LENode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const GTNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const GENode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const AndNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const OrNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const ReduceNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const CastNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const NotNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const SelectNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const RampNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const BroadcastNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const ShuffleNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const IntImmNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const FloatImmNode* op, ffi::reflection::AccessPath path) override;
  void VisitExpr_(const StringImmNode* op, ffi::reflection::AccessPath path) override;

  // Utility to call EnterDef/ExitDef.  Used in the implementation of
  // WithDef.
  template <typename T>
  class DefContext {
   public:
    DefContext(DefContext&& other) { swap(std::move(other)); }
    DefContext& operator=(DefContext&& other) {
      swap(std::move(other));
      return *this;
    }

    DefContext(const DefContext&) = delete;
    DefContext& operator=(const DefContext&) = delete;
    ~DefContext() noexcept(false) {
      // Checks performed when a definition goes out of scope may
      // raise an exception.  If the stack is already being unwound
      // due to another exception being thrown, this would cause a
      // segfault and terminate the program.  By checking that no
      // additional exceptions have been thrown between the
      // construction of the DefContext and the destruction, we avoid
      // this case and allow the first error to propagate upward.
      if (self_ && std::uncaught_exceptions() == uncaught_exceptions_) {
        self_->in_scope_definitions_.erase(obj_);
        self_->ExitDef(obj_, path_);
      }
    }

   private:
    friend class TIRVisitorWithPath;

    DefContext(TIRVisitorWithPath* self, T obj, ffi::reflection::AccessPath path)
        : self_(self), obj_(obj), path_(path), uncaught_exceptions_(std::uncaught_exceptions()) {
      self_->in_scope_definitions_.insert(obj_);
      self_->EnterDef(obj_, path_);
    }

    void swap(DefContext&& other) {
      std::swap(this->self_, other.self_);
      std::swap(this->obj_, other.obj_);
      std::swap(this->path_, other.path_);
      std::swap(this->uncaught_exceptions_, other.uncaught_exceptions_);
    }

    TIRVisitorWithPath* self_{nullptr};
    T obj_;
    ffi::reflection::AccessPath path_{ffi::reflection::AccessPath::Root()};
    int uncaught_exceptions_{-1};
  };

  // Utility to track the scope of a node's definition.
  template <typename T>
  DefContext<T> WithDef(T obj, ffi::reflection::AccessPath path) {
    return DefContext(this, obj, path);
  }

  /* \brief Utility to track the scope of a node's definition. */
  template <typename T>
  std::optional<DefContext<T>> WithDefIfUndefined(T obj, ffi::reflection::AccessPath path) {
    if (in_scope_definitions_.count(obj)) {
      return std::nullopt;
    } else {
      return WithDef(obj, path);
    }
  }

  std::vector<DefContext<Var>> WithMatchBufferDefs(Buffer buf, ffi::reflection::AccessPath path) {
    std::vector<DefContext<Var>> context;

    auto try_visit_implicit_var_def = [this, &context](const Expr& expr,
                                                       ffi::reflection::AccessPath path) {
      if (auto opt = expr.as<Var>()) {
        auto var = opt.value();
        if (auto var_def = WithDefIfUndefined(var, path)) {
          context.push_back(std::move(var_def).value());
        }
      }
    };
    auto try_visit_implicit_var_def_array = [&try_visit_implicit_var_def](
                                                const ffi::Array<PrimExpr>& arr,
                                                ffi::reflection::AccessPath path) {
      for (size_t i = 0; i < arr.size(); i++) {
        try_visit_implicit_var_def(arr[i], path->ArrayItem(i));
      }
    };

    try_visit_implicit_var_def(buf->data, path->Attr("data"));
    try_visit_implicit_var_def_array(buf->shape, path->Attr("shape"));
    try_visit_implicit_var_def_array(buf->strides, path->Attr("strides"));
    try_visit_implicit_var_def(buf->elem_offset, path->Attr("elem_offset"));

    return context;
  }

  std::unordered_set<ffi::ObjectRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> in_scope_definitions_;

  /*! \brief Scope stack for Bind variable definitions.
   *
   * Body-carrying statements (For, IfThenElse, etc.) push a new scope.
   * BindNode pushes its WithDef into the current scope.  When the
   * scope exits, all Bind defs are cleaned up automatically.
   */
  using BindScopeEntry = std::variant<DefContext<Var>, DefContext<Buffer>>;
  ScopeStack<std::vector<BindScopeEntry>> bind_scope_;
};

namespace {

template <typename DerivedVerifier>
class Verifier : protected TIRVisitorWithPath {
 public:
  template <typename TirNodeRef>
  static bool Verify(const TirNodeRef& node, bool assert_on_error) {
    DerivedVerifier verifier(assert_on_error);
    verifier(node);
    return !verifier.has_error_;
  }

 protected:
  explicit Verifier(bool assert_on_error) : assert_on_error_(assert_on_error) {}

  /* \brief Helper class to handle the bool-or-assert handles
   *
   * Each verifier can either return a boolean, or assert on failure.
   * To avoid needing to duplicate this logic at every step, the
   * Verify() method can be used.  Similar to `LOG(FATAL)` or
   * `LOG(DEBUG)`, it returns an object that can accept streamed
   * context information.
   *
   * If the error should be raised, then the context is collected
   * identically to `LOG(FATAL)`.  If a boolean is returned, or if the
   * condition passes, then the streamed context is discarded.
   *
   * Usage:
   *
   *     Verify(value == expected_value)
   *            << "ValueError: " << value
   *            << " was not the expected value of " << expected_value;
   */
  class VerifyStream {
   public:
    explicit VerifyStream(bool log_fatal) {
      if (log_fatal) {
        log_.emplace();
      }
    }

    VerifyStream(const VerifyStream&) = delete;
    VerifyStream& operator=(const VerifyStream&) = delete;
    VerifyStream(VerifyStream&& other) { std::swap(log_, other.log_); }
    VerifyStream& operator=(VerifyStream&& other) {
      std::swap(log_, other.log_);
      return *this;
    }

    template <typename T>
    VerifyStream& operator<<(T&& t) {
      if (log_.has_value()) {
        log_.value() << std::forward<T>(t);
      }
      return *this;
    }

    ~VerifyStream() noexcept(false) {
      if (log_.has_value()) {
        LOG(FATAL) << log_->str();
      }
    }

    std::optional<std::ostringstream> log_{std::nullopt};
  };

  // TODO(Lunderberg): Add the filename/linenum with
  // std::source_location when C++20 is available.
  VerifyStream Verify(bool condition) {
    has_error_ = has_error_ || !condition;
    return VerifyStream(!condition && assert_on_error_);
  }

  bool assert_on_error_;
  bool has_error_{false};
};

}  // namespace

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_IR_TIR_VISITOR_WITH_PATH_H_
