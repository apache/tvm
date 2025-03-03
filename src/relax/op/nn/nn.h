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
 * \file nn.h
 * \brief The functions to make Relax neural network operator calls.
 */

#ifndef TVM_RELAX_OP_NN_NN_H_
#define TVM_RELAX_OP_NN_NN_H_

#include <tvm/relax/attrs/nn.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Quick helper macro to
 * - expose a make-function interface which construct the call node.
 * - register op to the registry.
 * \param OpName The name of operator to register.
 * \param OpRegName The identifier of the operator in the registry.
 * \param RequireFloatDtype A boolean indicating if the input is required to have float dtype.
 */
#define RELAX_REGISTER_UNARY_NN_OP_AND_IMPL(OpName, OpRegName, RequireFloatDtype) \
  RELAX_REGISTER_UNARY_OP(OpRegName).set_attr<FInferStructInfo>(                  \
      "FInferStructInfo", InferStructInfoUnaryArith<RequireFloatDtype>);          \
  RELAX_UNARY_OP_INTERFACE(OpName, OpRegName);

/*! \brief Rectified linear unit. */
Expr relu(Expr data);

/*! \brief Leaky rectified linear unit. */
Expr leakyrelu(Expr data, double alpha);

/*! \brief Gaussian Error Linear Units function. */
Expr gelu(Expr data);

/*! \brief Gaussian Error Linear Units function approximated by tanh. */
Expr gelu_tanh(Expr data);

/*! \brief Sigmoid Linear Unit function. */
Expr silu(Expr data);

/*! \brief Softmax function. */
Expr softmax(Expr data, int axis);

/*! \brief LogSoftmax function. */
Expr log_softmax(Expr data, int axis);

/*! \brief Compute batch normalization. */
Expr batch_norm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                int axis, double epsilon, bool center, bool scale, double momentum);

/*! \brief Compute layer normalization. */
Expr layer_norm(Expr data, Expr gamma, Expr beta, Array<Integer> axes, double epsilon, bool center,
                bool scale);

/*! \brief Compute group normalization. */
Expr group_norm(Expr data, Expr gamma, Expr beta, int num_groups, int channel_axis,
                Array<Integer> axes, double epsilon, bool center, bool scale);

/*! \brief Compute root mean square normalization. */
Expr rms_norm(Expr data, Expr weight, Array<Integer> axes, double epsilon);

/*!
 * \brief Applies the dropout operation to the input tensor.
 * \param data The input data to the operator.
 * \param rate The probability for an element to be reset to 0.
 * \return A Tuple of two tensors.
 * The first one is the original tensor and the second one is a
 * mask tensor (1.0 where element not dropped, 0.0 where dropped)
 */
Expr dropout(Expr data, double rate);

/*! \brief CrossEntropy with logits. */
Expr cross_entropy_with_logits(Expr predictions, Expr labels);

/*! \brief Negative log likelihood loss. */
Expr nll_loss(Expr predictions, Expr targets, Optional<Expr> weights, String reduction,
              int ignore_index);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_NN_H_
