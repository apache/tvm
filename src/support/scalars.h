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
 * \file src/support/scalars.h
 * \brief Helpers for converting between scalars in native, text, TIR immediate and NDArray forms.
 */

#ifndef TVM_SUPPORT_SCALARS_H_
#define TVM_SUPPORT_SCALARS_H_

#include <string>
#include <utility>

#include "tvm/ir/expr.h"
#include "tvm/relay/expr.h"
#include "tvm/runtime/ndarray.h"

namespace tvm {
namespace support {

/*! \brief Returns true if \p constant_node is a float/int/bool scalar. */
bool IsSimpleScalar(const relay::ConstantNode* constant_node);

/*! \brief Returns NDArray 'scalar' for given TIR immediate. */
runtime::NDArray IntImmToNDArray(const IntImm& int_imm);
runtime::NDArray FloatImmToNDArray(const FloatImm& float_imm);
runtime::NDArray BoolToNDArray(bool value);

/*! \brief Returns Relay literal text for NDArray 'scalar'. */
std::string NDArrayScalarToString(const runtime::NDArray& data);

/*! \brief Returns Relay literal text for given TIR immediate. */
std::string IntImmToString(const IntImm& int_imm);
std::string FloatImmToString(const FloatImm& float_imm);
std::string BoolToString(bool value);

/*!
 * \brief Returns TIR immediate for given value and width. Boolean will be true if value
 * was clipped in order to stay within range for width. However:
 *  - we ignore underflow
 *  - we don't currently check for float16 limits.
 */
std::pair<IntImm, bool> ValueToIntImm(int64_t value, int width);
std::pair<FloatImm, bool> ValueToFloatImm(double value, int width);

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_SCALARS_H_
