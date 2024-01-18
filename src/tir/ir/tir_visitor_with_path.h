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
#include <utility>

namespace tvm {
namespace tir {

/*! \brief Visit TIR while tracking the ObjectPath */
class TIRVisitorWithPath : protected ExprFunctor<void(const PrimExpr&, ObjectPath)>,
                           protected StmtFunctor<void(const Stmt&, ObjectPath)> {
 public:
  template <typename TObjectRef>
  void operator()(TObjectRef&& obj) {
    Visit(std::forward<TObjectRef>(obj), ObjectPath::Root());
  }

 protected:
  // Delegate to ExprFunctor::VisitExpr for PrimExpr, and any subclasses
  inline void Visit(const PrimExpr& obj, ObjectPath path) { VisitExpr(obj, path); }
  // Delegate to ExprFunctor::VisitStmt for Stmt, and any subclasses
  inline void Visit(const Stmt& obj, ObjectPath path) { VisitStmt(obj, path); }

  // Visitors for TIR constructs that are neither PrimExpr nor Stmt
  virtual void Visit(const IRModule& obj, ObjectPath path);
  virtual void Visit(const PrimFunc& obj, ObjectPath path);
  virtual void Visit(const GlobalVar& obj, ObjectPath path) {}
  virtual void Visit(const Range& obj, ObjectPath path);
  virtual void Visit(const Buffer& obj, ObjectPath path);
  virtual void Visit(const BufferRegion& obj, ObjectPath path);
  virtual void Visit(const MatchBufferRegion& obj, ObjectPath path);
  virtual void Visit(const IterVar& obj, ObjectPath path);

  // Called when entering/exiting the scope of a GlobalVar definition.
  virtual void EnterDef(const GlobalVar& var, ObjectPath path) {}
  virtual void ExitDef(const GlobalVar& var, ObjectPath path) {}

  // Called when entering/exiting the scope of a tir::Var definition.
  virtual void EnterDef(const Var& var, ObjectPath path) {}
  virtual void ExitDef(const Var& var, ObjectPath path) {}

  // Called when entering/exiting the scope of an IterVar definition.
  // By default, visits the `Range IterVarNode::dom`, then enters the
  // scope of the internal `tir::Var`.
  virtual void EnterDef(const IterVar& var, ObjectPath path);
  virtual void ExitDef(const IterVar& var, ObjectPath path);

  // Called when entering/exiting the scope of a Buffer definition.
  // By default, visits the buffer's data pointer, shape, strides, and
  // elem_offset, which must be defined prior to defining the Buffer.
  virtual void EnterDef(const Buffer& buffer, ObjectPath path);
  virtual void ExitDef(const Buffer& buffer, ObjectPath path);

  // Utility to visit an array of nodes
  template <typename T>
  inline void Visit(const Array<T>& arr, ObjectPath path) {
    for (size_t i = 0; i < arr.size(); i++) {
      Visit(arr[i], path->ArrayIndex(i));
    }
  }

  // Utility to visit an optional node nodes
  template <typename T>
  inline void Visit(const Optional<T>& opt, ObjectPath path) {
    if (opt) {
      Visit(opt.value(), path);
    }
  }

  using StmtFunctor::VisitStmt;
  void VisitStmt_(const AttrStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const IfThenElseNode* op, ObjectPath path) override;
  void VisitStmt_(const LetStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const ForNode* op, ObjectPath path) override;
  void VisitStmt_(const WhileNode* op, ObjectPath path) override;
  void VisitStmt_(const AllocateNode* op, ObjectPath path) override;
  void VisitStmt_(const AllocateConstNode* op, ObjectPath path) override;
  void VisitStmt_(const DeclBufferNode* op, ObjectPath path) override;
  void VisitStmt_(const BufferStoreNode* op, ObjectPath path) override;
  void VisitStmt_(const BufferRealizeNode* op, ObjectPath path) override;
  void VisitStmt_(const AssertStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const ProducerStoreNode* op, ObjectPath path) override;
  void VisitStmt_(const ProducerRealizeNode* op, ObjectPath path) override;
  void VisitStmt_(const PrefetchNode* op, ObjectPath path) override;
  void VisitStmt_(const SeqStmtNode* op, ObjectPath path) override;
  void VisitStmt_(const EvaluateNode* op, ObjectPath path) override;
  void VisitStmt_(const BlockNode* op, ObjectPath path) override;
  void VisitStmt_(const BlockRealizeNode* op, ObjectPath path) override;

  using ExprFunctor::VisitExpr;
  void VisitExpr_(const VarNode* op, ObjectPath path) override;
  void VisitExpr_(const SizeVarNode* op, ObjectPath path) override;
  void VisitExpr_(const BufferLoadNode* op, ObjectPath path) override;
  void VisitExpr_(const ProducerLoadNode* op, ObjectPath path) override;
  void VisitExpr_(const LetNode* op, ObjectPath path) override;
  void VisitExpr_(const CallNode* op, ObjectPath path) override;
  void VisitExpr_(const AddNode* op, ObjectPath path) override;
  void VisitExpr_(const SubNode* op, ObjectPath path) override;
  void VisitExpr_(const MulNode* op, ObjectPath path) override;
  void VisitExpr_(const DivNode* op, ObjectPath path) override;
  void VisitExpr_(const ModNode* op, ObjectPath path) override;
  void VisitExpr_(const FloorDivNode* op, ObjectPath path) override;
  void VisitExpr_(const FloorModNode* op, ObjectPath path) override;
  void VisitExpr_(const MinNode* op, ObjectPath path) override;
  void VisitExpr_(const MaxNode* op, ObjectPath path) override;
  void VisitExpr_(const EQNode* op, ObjectPath path) override;
  void VisitExpr_(const NENode* op, ObjectPath path) override;
  void VisitExpr_(const LTNode* op, ObjectPath path) override;
  void VisitExpr_(const LENode* op, ObjectPath path) override;
  void VisitExpr_(const GTNode* op, ObjectPath path) override;
  void VisitExpr_(const GENode* op, ObjectPath path) override;
  void VisitExpr_(const AndNode* op, ObjectPath path) override;
  void VisitExpr_(const OrNode* op, ObjectPath path) override;
  void VisitExpr_(const ReduceNode* op, ObjectPath path) override;
  void VisitExpr_(const CastNode* op, ObjectPath path) override;
  void VisitExpr_(const NotNode* op, ObjectPath path) override;
  void VisitExpr_(const SelectNode* op, ObjectPath path) override;
  void VisitExpr_(const RampNode* op, ObjectPath path) override;
  void VisitExpr_(const BroadcastNode* op, ObjectPath path) override;
  void VisitExpr_(const ShuffleNode* op, ObjectPath path) override;
  void VisitExpr_(const IntImmNode* op, ObjectPath path) override;
  void VisitExpr_(const FloatImmNode* op, ObjectPath path) override;
  void VisitExpr_(const StringImmNode* op, ObjectPath path) override;
  void VisitExpr_(const AnyNode* op, ObjectPath path) override;

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
        self_->ExitDef(obj_, path_);
      }
    }

   private:
    friend class TIRVisitorWithPath;

    DefContext(TIRVisitorWithPath* self, T obj, ObjectPath path)
        : self_(self), obj_(obj), path_(path), uncaught_exceptions_(std::uncaught_exceptions()) {
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
    ObjectPath path_{ObjectPath::Root()};
    int uncaught_exceptions_{-1};
  };

  // Utility to track the scope of a node's definition.
  template <typename T>
  DefContext<T> WithDef(T obj, ObjectPath path) {
    return DefContext(this, obj, path);
  }
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_TIR_VISITOR_WITH_PATH_H_
