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
 * \file src/contrib/ethosu/cascader/common.h
 * \brief Common functions used in the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_COMMON_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_COMMON_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>

#include <functional>
#include <numeric>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*!
 * \brief Make a tvm::Array<Integer> from an int vector.
 * \param vec The int vector.
 * \return The Integer Array.
 * \note Array<Integer>(std::vector<int>) doesn't work as this implicit
 * type conversion fails. This is why this helper is required.
 */
inline Array<Integer> make_array(const std::vector<int>& vec) {
  Array<Integer> arr;
  arr.resize(vec.size());
  for (unsigned int i = 0; i < vec.size(); ++i) {
    arr.Set(i, Integer(vec[i]));
  }
  return arr;
}

/*!
 * \brief Make a tvm::Array<Integer> from a size_t vector.
 * \param vec The size_t vector.
 * \return The Integer Array.
 * \note Array<Integer>(std::vector<size_t>) doesn't work as this implicit
 * type conversion fails. This is why this helper is required.
 */
inline Array<Integer> make_array(const std::vector<size_t>& vec) {
  Array<Integer> arr;
  arr.resize(vec.size());
  for (unsigned int i = 0; i < vec.size(); ++i) {
    arr.Set(i, Integer(vec[i]));
  }
  return arr;
}

/*!
 * \brief Make a tvm::Array<IntImm> from an int64_t vector.
 * \param vec The int64_t vector.
 * \return The IntImm Array.
 * \note Array<IntImm>(std::vector<int64_t>) doesn't work as this implicit
 * type conversion fails. This is why this helper is required.
 */
inline Array<IntImm> make_array(const std::vector<int64_t>& vec) {
  Array<IntImm> arr;
  arr.resize(vec.size());
  for (unsigned int i = 0; i < vec.size(); ++i) {
    arr.Set(i, IntImm(DataType::Int(64), vec[i]));
  }
  return arr;
}

/*!
 * \brief Make a tvm::Array<FloatImm> from an float vector.
 * \param vec The float vector.
 * \return The FloatImm Array.
 */
inline Array<FloatImm> make_array(const std::vector<float>& vec) {
  Array<FloatImm> arr;
  arr.resize(vec.size());
  for (unsigned int i = 0; i < vec.size(); ++i) {
    arr.Set(i, FloatImm(DataType::Float(32), static_cast<double>(vec[i])));
  }
  return arr;
}

/*!
 * \brief Calculate the ceil of an Integer division
 * \param dividend The dividend of the division
 * \param divisor The divisor of the division
 * \return The quotient
 */
inline int round_up_divide(int dividend, int divisor) {
  return dividend / divisor + (dividend % divisor != 0);
}

/*!
 * \brief Make a vector from a tvm::Array.
 * \param arr The Array.
 * \return The vector.
 */
template <typename T, typename tvm_T>
inline std::vector<T> make_vector(const Array<tvm_T>& arr) {
  std::vector<T> vec(arr.size());
  for (unsigned int i = 0; i < arr.size(); ++i) {
    vec[i] = arr[i]->value;
  }
  return vec;
}

/*!
 * \brief Create a combined hash.
 * \param seed The current hash value.
 * \param v The value to combine into the hash.
 * \return The combined hash.
 */
template <class T>
inline void hash_combine(std::size_t* seed, T const& v) {
  *seed ^= std::hash<T>()(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
}

/*!
 * \brief Hash a vector.
 * \param vec The vector to hash.
 * \return The hash.
 */
template <class T>
inline std::size_t hash_vector(const std::vector<T>& vec) {
  std::size_t seed = vec.size();
  for (const auto& elem : vec) {
    hash_combine(&seed, elem);
  }
  return seed;
}

template <class T>
inline T mul_reduce(const std::vector<T>& vec) {
  return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<T>());
}

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_COMMON_H_
