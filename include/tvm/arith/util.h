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
 * \file tvm/arith/util.h
 * \brief Utils for arithmetic analysis.
 */
#ifndef TVM_ARITH_UTIL_H_
#define TVM_ARITH_UTIL_H_

#include <cstdint>
#include <tuple>

namespace tvm {
/*! \brief namespace of arithmetic analysis. */
namespace arith {

/*!
 * \brief Calculate the extended greatest common divisor for two values.
 *        See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm.
 * \param a an integer number
 * \param b an integer number
 * \return 3 integers (div, m, n) where div = gcd(a, b) and a*m + b*n = div
 */
std::tuple<int64_t, int64_t, int64_t> xgcd(int64_t a, int64_t b);

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_UTIL_H_
