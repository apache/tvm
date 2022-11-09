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
 * \file topi/einsum.h
 * \brief Einstein summation op
 */
#ifndef TVM_TOPI_EINSUM_H_
#define TVM_TOPI_EINSUM_H_

#define LABELRANGE 128
#define NPY_MAXDIMS 16
#define NPY_MAXARGS 16

#include <tvm/te/operation.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/detail/ravel_unravel.h>
#include <tvm/topi/detail/tensor_utils.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <bitset>
#include <iterator>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace topi {

using namespace tvm::te;
using namespace topi::detail;

/*!
 * \brief Compute the shape of the output.
 * \param subscripts input subscripts.
 * \param operands operand tensors.
 *
 * \return the shape of the output.
 */
Array<PrimExpr> InferEinsumShape(const std::string& subscripts,
                                 const std::vector<Array<PrimExpr>>& operands);

/*!
 * \brief Evaluates the Einstein summation convention on the operands.
 *
 * \param subscripts_str Specifies the subscripts for summation as comma separated list of
 * subscript labels.
 * \param inputs Arrays for the operation.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return The calculation based on the Einstein summation convention.
 */
Tensor einsum(const std::string& subscripts_str, const Array<Tensor> inputs,
              std::string name = "T_einsum", std::string tag = kEinsum);

struct EinsumEquation {
  /*!
   * \brief Create EinsumEquation from a string.
   * The result will be converted to the explicit mode of Einsum if it is in implicit mode.
   * \return The created EinsumEquation.
   */
  static EinsumEquation FromString(const std::string& equation);
  using Label = char;
  using Subscript = std::vector<Label>;
  // Special label value for ellipsis. The value is chosen to be less than any other letters so make
  // sorting easier.
  static constexpr Label kEllipsis = '\0';
  // The input subscripts for each operand of the Einsum operator.
  std::vector<Subscript> inputs;
  // The output subscript of the Einsum equation.
  Subscript output;
};

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_EINSUM_H_
