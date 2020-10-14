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
 * \file ad_utils.h
 * \brief Helper utilities to implement auto-differentiation.
 */
#ifndef TVM_TE_AUTODIFF_AD_UTILS_H_
#define TVM_TE_AUTODIFF_AD_UTILS_H_

#include <tvm/arith/int_solver.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace te {

/*!
 * \brief Clone iter vars and return both the new vars and the substitution from old to new.
 *
 * \param vars The original iter vars.
 * \return A pair containing the array of new iter vars and the map from old vars to new ones.
 */
std::pair<Array<IterVar>, Map<Var, PrimExpr>> CloneIterVars(const Array<IterVar>& vars);

/*!
 * \brief Clone reduction by cloning the axis variables.
 * \param expr A reduction expr to clone. Non-reduction expressions are left intact.
 */
PrimExpr CloneReduction(const PrimExpr& expr);

/*!
 * \brief Create a tensor from an expression. The expression may be a reduction, in which
 *  case its body will be correctly duplicated if it is a multi-valued reduction.
 *
 * \param expr The expr which will be the tensor's body.
 * \param axis The input variables with ranges.
 * \param name The tensor's name.
 * \param tag The tensor's tag.
 * \param attrs The tensor's attrs.
 * \param clone_axis Whether to clone the given axis and perform substitution.
 * \return A tensor.
 */
Tensor TensorFromExpr(const PrimExpr& expr, const Array<IterVar>& axis,
                      const std::string& name = "tensor", const std::string& tag = "",
                      const Map<String, ObjectRef>& attrs = {}, bool clone_axis = true);

Tensor TransformTensorBody(
    const Tensor& tensor,
    const std::function<PrimExpr(const PrimExpr&, const Array<IterVar>&)>& func);

Tensor TransformTensorBody(const Tensor& tensor,
                           const std::function<PrimExpr(const PrimExpr&)>& func);

/*!
 * \brief Inline tensors access recursively.
 *
 *  This function will inline tensors recursively until it reaches a tensor which is impossible to
 *  inline (a reduction if \p inline_reductions is false, a non-compute tensor, a tensor which is
 *  not from \p inlineable). It won't descend into non-inlinable tensors' bodies.
 *
 * \param tensor The tensor whose body to transform.
 * \param inlineable A list of tensors which are allowed to be inlined. If empty, try
 *  to inline all tensors.
 * \param inline_reductions Whether to inline reductions (this may result in top-level reduction
 *  nodes).
 *
 * \return An inlined tensor
 */
TVM_DLL Tensor InlineTensorAccess(const Tensor& tensor,
                                  const Array<Tensor>& inlineable = Array<Tensor>(),
                                  bool inline_reductions = false);

/*!
 * \brief Inline tensors access at the tail.
 * \param tensor The tensor whose body to transform.
 * \return An inlined tensor
 */
TVM_DLL Tensor InlineTailTensorAccess(const Tensor& tensor);

/*!
 * \brief Simplify an iteration domain.
 *
 *  An iteration domain is basically an array of variables and a condition. The function will do the
 *  following:
 *  - Replace div and mod operations with new variables (optional).
 *  - Extract (in)equalities from the condition.
 *  - Perform Fourier-Motzkin elimination.
 *  - Shear the domain of iteration (e.g. if `y <= x <= y + 2` then x will be replaced with `y + d`
 *    where `d` is a new variable such that `0 <= d <= 2`).
 *  - Remove redundant variables.
 *  - Infer new variable ranges (hopefully more precise).
 *
 * \param iter_domains The original domain.
 * \param eliminate_div_mod Whether to eliminate div and mod by introducing new variables.
 */
TVM_DLL arith::IntConstraintsTransform SimplifyDomain(const arith::IntConstraints& iter_domains,
                                                      bool eliminate_div_mod = true);

/*!
 * \brief Perform lifting of conditions of being possible to be non-zero together with
 *  applying some transformations like simplifying the reduction domain. Works only with
 *  this particular tensor's body, i.e. doesn't perform inlining.
 *
 * \param tensor The original tensor;
 * \param vranges Optional map from free variables to their value ranges.
 * \return An optimized tensor.
 */
TVM_DLL Tensor RemoveJacobianAndLiftNonzeroCond(const Tensor& tensor,
                                                const Map<Var, Range>& vranges = Map<Var, Range>());

}  // namespace te
}  // namespace tvm
#endif  // TVM_TE_AUTODIFF_AD_UTILS_H_
