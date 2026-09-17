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
 * \brief Dialect entry point for shared expression dispatch.
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
  using Parent = tvm::ExprFunctor<R(const Expr&, Args...)>;

 public:
  using Parent::Dispatch_;

  /*! \brief Construct a functor with the inherited expression hooks. */
  ExprFunctor() : Parent(GlobalVTable()) {}
  /*! \brief Destroy through the dialect functor base. */
  virtual ~ExprFunctor() = default;

 protected:
  using Parent::SetDispatch;
  using typename Parent::VTable;

  /*! \brief Construct with a finalized table that outlives the functor. */
  explicit ExprFunctor(const VTable* vtable) : Parent(vtable) {}

  /*! \brief Initialize the inherited expression dispatch. */
  static void InitVTable(VTable* vtable) { Parent::InitVTable(vtable); }

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

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_EXPR_FUNCTOR_H_
