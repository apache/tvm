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
 *  Copyright (c) 2019 by Contributors
 * \file modular_set.h
 * \brief Modular set analysis
 */
#ifndef TVM_ARITHMETIC_MODULAR_SET_H_
#define TVM_ARITHMETIC_MODULAR_SET_H_

#include <tvm/arithmetic.h>

namespace tvm {
namespace arith {

/*!
 * \brief Take GCD of a and b.
 * \param a The first operand.
 * \param b The second operand.
 * \return The result.
 */
int64_t ZeroAwareGCD(int64_t a, int64_t b);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_MODULAR_SET_H_
