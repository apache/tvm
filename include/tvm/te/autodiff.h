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
 * \file tvm/te/autodiff.h
 * \brief Automatic differentiation of tensor expressions.
 */

#ifndef TVM_TE_AUTODIFF_H_
#define TVM_TE_AUTODIFF_H_

#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>

#include "tensor.h"

namespace tvm {
/*! \brief Tensor expression language DSL. */
namespace te {

/*!
 * \brief Take the derivative of the expression with respect to the given variable.
 * \param expr The expression to differentiate.
 * \param var The variable to differentiate with respect to.
 * \return The expression for the derivative.
 */
PrimExpr Derivative(const PrimExpr& expr, const Var& var);

/*!
 * \brief Get the tensor representing the Jacobian of the output with respect to the input.
 *
 *  Note that if \p output depends on \p input indirectly (by using some other tensor
 *  depending on \p input), this dependency won't contribute to the resulting Jacobian.
 *  For such cases use the function ::Gradient.
 *
 * \param output The tensor to differentiate.
 * \param input The input tensor, which \p output should directly use.
 * \return The tensor representing the Jacobian of shape `output.shape + input.shape`.
 */
Tensor Jacobian(const Tensor& output, const Tensor& input);

/*!
 * \brief The building block for reverse-mode AD.
 *
 *  Differentiate \p output wrt \p input and multiply the result by \p head on the left using tensor
 *  dot product. \p input must be an immediate dependency of \p output (must be called from within
 *  the body of \p output). That is, the function will compute one summand of the adjoint for \p
 * input given the adjoint for \p output (which is called \p head here).
 *
 * \param output The tensor to differentiate.
 * \param input The input tensor, which \p output should directly use.
 * \param head The adjoint of \p output. Must be of shape `prefix + output.shape`
 * \return The tensor of shape `prefix + input.shape`
 *         representing the partial adjoint of \p input wrt one of its consumers (output)
 */
Tensor VectorJacobianProduct(const Tensor& output, const Tensor& input, const Tensor& head);

/*!
 * \brief Perform reverse mode automatic differentiation.
 *
 *  Each item of the `result` field of the result is an adjoint for the corresponding item of
 *  \p inputs, i.e. \p head multiplied by the Jacobian of \p output with respect to the
 *  corresponding item of \p inputs.
 *
 * \param output The tensor to differentiate.
 * \param inputs The array of input tensors. When the array is empty, will perform differentiation
 *               wrt all tensors the output depends on.
 * \param head The adjoint of the output, in other words, some tensor, by which the Jacobians
 *             will be multiplied (using tensordot axes=`output.shape`).
 *             Its shape must be of the form `prefix + output.shape`. If the null pointer is
 * provided, the identity tensor of shape `output.shape + output.shape` will be used. \return An
 * array of adjoints corresponding to \p inputs.
 */
TVM_DLL Array<Tensor> Gradient(const Tensor& output, const Array<Tensor>& inputs,
                               const Tensor& head = Tensor());

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_AUTODIFF_H_
