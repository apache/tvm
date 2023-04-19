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
 * \file tvm/relax/training/utils.h
 * \brief Utility classes and functions for relax training.
 */
#ifndef TVM_RELAX_TRAINING_UTILS_H_
#define TVM_RELAX_TRAINING_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace transform {

/*!
 * \brief Append the loss function to the backbone function specified by `func_name`. Generally, the
 * loss function is generated by instances of `relax.training.Loss`.
 *
 * The backbone function and the loss function should satisfy a few restrictions:
 * - Both backbone and loss should contain exactly one DataflowBlock.
 * - Backbone should return either one Var, or a tuple of Vars
 * - Loss should return a scalar(0-dim Tensor) Var
 *
 * The appended result contains only one DataflowBlock containing all bindings in backbone and loss.
 *
 * \param func_name The name of the backbone function in the IRModule.
 * \param loss_function The loss function.
 * \param num_backbone_outputs Specify the number of `prediction_outputs` of the backbone function.
 * Default: 1.
 * \param new_func_name Specify the name of the appended result. If is is not specified, the name
 * will be `func_name + "_loss"`.
 * \return The Pass.
 */
TVM_DLL Pass AppendLoss(String func_name, Function loss_function, int num_backbone_outputs = 1,
                        Optional<String> new_func_name = NullOpt);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRAINING_UTILS_H_
