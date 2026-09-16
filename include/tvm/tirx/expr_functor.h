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
 * \file tvm/tirx/expr_functor.h
 *
 * \brief Functors for tirx expressions.
 */
#ifndef TVM_TIR_EXPR_FUNCTOR_H_
#define TVM_TIR_EXPR_FUNCTOR_H_

#include <tvm/ir/expr_functor.h>
#include <tvm/tirx/buffer_region.h>

#include <utility>

namespace tvm {
namespace tirx {

/*!
 * \brief Shared expression dispatch extended with the TIRx BufferRegion hook.
 *
 * Override Dispatch_ for a node type or DispatchDefault_ for default behavior.
 * Core expression hooks, result types, argument forwarding and registered-ancestor
 * dispatch are inherited from tvm::ExprFunctor. This functor does not recurse
 * automatically. Derived dialect extensions can initialize a fresh inherited
 * table with InitVTable and register additional hooks with SetDispatch.
 * \tparam FType A function signature of the form R(const Expr&, Args...).
 */
template <typename FType>
class ExprFunctor;

template <typename R, typename... Args>
class ExprFunctor<R(const Expr&, Args...)> : public tvm::ExprFunctor<R(const Expr&, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr&, Args...)>;
  using Parent = tvm::ExprFunctor<R(const Expr&, Args...)>;

 public:
  using Parent::Dispatch_;

  /*! \brief Construct a functor with the inherited core and TIRx hooks. */
  ExprFunctor() : Parent(GlobalVTable()) {}
  /*! \brief Destroy through the dialect functor base. */
  virtual ~ExprFunctor() = default;

  virtual R Dispatch_(const BufferRegionNode* node, Args... args) {
    return this->DispatchDefault_(node, std::forward<Args>(args)...);
  }

 protected:
  using Parent::SetDispatch;
  using typename Parent::VTable;

  /*! \brief Construct with a finalized table that outlives the functor. */
  explicit ExprFunctor(const VTable* vtable) : Parent(vtable) {}

  /*! \brief Initialize inherited dispatch and add the dialect-only hook. */
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    Parent::template SetDispatch<TSelf, BufferRegionNode>(vtable);
  }

 private:
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class TVM_DLL ExprMutator : protected ExprFunctor<Expr(const Expr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::Dispatch;
  /*! \brief Visit a primitive expression and verify that it remains primitive. */
  PrimExpr VisitPrimExpr(const PrimExpr& expr) { return Dispatch(expr).as_or_throw<PrimExpr>(); }
  // list of functions to override.
  Expr Dispatch_(const VarNode* op) override;
  Expr Dispatch_(const TensorLoadNode* op) override;
  Expr Dispatch_(const OpaqueExprNode* op) override;
  Expr Dispatch_(const BufferRegionNode* op) override;
  Expr Dispatch_(const TupleNode* op) override;
  Expr Dispatch_(const TupleGetItemNode* op) override;
  Expr Dispatch_(const prim::LetNode* op) override;
  Expr Dispatch_(const CallNode* op) override;
  Expr Dispatch_(const prim::AddNode* op) override;
  Expr Dispatch_(const prim::SubNode* op) override;
  Expr Dispatch_(const prim::MulNode* op) override;
  Expr Dispatch_(const prim::DivNode* op) override;
  Expr Dispatch_(const prim::ModNode* op) override;
  Expr Dispatch_(const prim::FloorDivNode* op) override;
  Expr Dispatch_(const prim::FloorModNode* op) override;
  Expr Dispatch_(const prim::MinNode* op) override;
  Expr Dispatch_(const prim::MaxNode* op) override;
  Expr Dispatch_(const prim::EQNode* op) override;
  Expr Dispatch_(const prim::NENode* op) override;
  Expr Dispatch_(const prim::LTNode* op) override;
  Expr Dispatch_(const prim::LENode* op) override;
  Expr Dispatch_(const prim::GTNode* op) override;
  Expr Dispatch_(const prim::GENode* op) override;
  Expr Dispatch_(const prim::AndNode* op) override;
  Expr Dispatch_(const prim::OrNode* op) override;
  Expr Dispatch_(const prim::CastNode* op) override;
  Expr Dispatch_(const prim::NotNode* op) override;
  Expr Dispatch_(const prim::SelectNode* op) override;
  Expr Dispatch_(const prim::RampNode* op) override;
  Expr Dispatch_(const prim::BroadcastNode* op) override;
  Expr Dispatch_(const prim::ShuffleNode* op) override;
  Expr Dispatch_(const IntImmNode* op) override;
  Expr Dispatch_(const FloatImmNode* op) override;
  Expr Dispatch_(const prim::StringImmNode* op) override;
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_EXPR_FUNCTOR_H_
