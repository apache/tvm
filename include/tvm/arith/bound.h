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
 * \file tvm/arith/bound.h
 * \brief Bound deducers.
 */
#ifndef TVM_ARITH_BOUND_H_
#define TVM_ARITH_BOUND_H_

#include <tvm/node/container.h>
#include <tvm/ir/expr.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include <unordered_map>

namespace tvm {
// forward delcare Tensor
namespace te {
class Tensor;
}
namespace arith {

using tir::Var;
using tir::VarNode;
using tir::Domain;
using tir::Stmt;

/*!
 * \brief Deduce the bound of the target variable in a expression,
 *  give the domain of each variables. Return undefined IntSet to
 *  represent failure.
 *
 * \note The returned set may be smaller than set that
 *       contains all possible values of v that satisfies the bound.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound must implies e for all value in relax_map
 * \return An integer set that always satisfies the condition.
 */
IntSet DeduceBound(PrimExpr v, PrimExpr cond,
                   const Map<Var, IntSet>& hint_map,
                   const Map<Var, IntSet>& relax_map);
/*!
 * \brief Same as DeduceBound with  unordered_map signature.
 *
 * \param v The target variable to be deduced.
 * \param cond The conditional expression.
 * \param hint_map The domain of variable, used to help deduce.
 * \param relax_map The domain of each variable, used to relax the domain,
 *        The deduce bound mush implies e for all value in relax_map
 * \return An integer set that always satisfies the condition.
 */
IntSet DeduceBound(PrimExpr v, PrimExpr cond,
                   const std::unordered_map<const VarNode*, IntSet>& hint_map,
                   const std::unordered_map<const VarNode*, IntSet>& relax_map);

/*!
 * \brief Infer a regular domain that covers all the calls or provides within the given statement.
 * \param body The given statement.
 * \param buffer The buffer to check the access info.
 * \param consider_loads If loads are considered.
 * \param consider_stores If stores are considered.
 * \return The domain that covers all the calls or provides within the given statement.
 */
Domain DomainTouched(const Stmt& body,
                     const tir::Buffer& buffer,
                     bool consider_loads,
                     bool consider_stores);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_BOUND_H_
