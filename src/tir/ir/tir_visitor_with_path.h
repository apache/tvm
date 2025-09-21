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
 * \file tir/ir/tir_visitor_with_path.h
 * \brief Provide a TIR visitor that tracks the current location
 */
#ifndef TVM_TIR_IR_TIR_VISITOR_WITH_PATH_H_
#define TVM_TIR_IR_TIR_VISITOR_WITH_PATH_H_

#include <tvm/ir/module.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

#include <exception>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {

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
  inline void Visit(const PrimExpr& obj, ffi::reflection::AccessPath path) { VisitExpr(obj, path); }
  // Delegate to ExprFunctor::VisitStmt for Stmt, and any subclasses
  inline void Visit(const Stmt& obj, ffi::reflection::AccessPath path) { VisitStmt(obj, path); }

  // Visitors for TIR constructs that are neither PrimExpr nor Stmt
  virtual void Visit(const IRModule& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const PrimFunc& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const GlobalVar& obj, ffi::reflection::AccessPath path) {}
  virtual void Visit(const Range& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const Buffer& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const BufferRegion& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const MatchBufferRegion& obj, ffi::reflection::AccessPath path);
  virtual void Visit(const IterVar& obj, ffi::reflection::AccessPath path);

  // Called when entering/exiting the scope of a GlobalVar definition.
  virtual void EnterDef(const GlobalVar& var, ffi::reflection::AccessPath path) {}
  virtual void ExitDef(const GlobalVar& var, ffi::reflection::AccessPath path) {}

  // Called when entering/exiting the scope of a tir::Var definition.
  virtual void EnterDef(const Var& var, ffi::reflection::AccessPath path) {}
  virtual void ExitDef(const Var& var, ffi::reflection::AccessPath path) {}

  // Called when entering/exiting the scope of an IterVar definition.
  // By default, visits the `Range IterVarNode::dom`, then enters the
  // scope of the internal `tir::Var`.
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
  void VisitStmt_(const AttrStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const IfThenElseNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const LetStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const ForNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const WhileNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AllocateNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AllocateConstNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const DeclBufferNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BufferStoreNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BufferRealizeNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const AssertStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const SeqStmtNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const EvaluateNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BlockNode* op, ffi::reflection::AccessPath path) override;
  void VisitStmt_(const BlockRealizeNode* op, ffi::reflection::AccessPath path) override;

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

    auto try_visit_implicit_var_def = [this, &context](const PrimExpr& expr,
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

  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> in_scope_definitions_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_TIR_VISITOR_WITH_PATH_H_
