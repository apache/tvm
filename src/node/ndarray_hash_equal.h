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
#ifndef TVM_NODE_NDARRAY_HASH_EQUAL_H_
#define TVM_NODE_NDARRAY_HASH_EQUAL_H_

#include <tvm/runtime/ndarray.h>

namespace tvm {

class SEqualReducer;
class SHashReducer;

/*!
 * \brief Test two NDArrays for equality.
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param equal A Reducer class to reduce the structural equality result of two objects.
 * See tvm/node/structural_equal.h.
 * \param compare_data Whether or not to consider ndarray raw data in the equality testing.
 * \return The equality testing result.
 */
bool NDArrayEqual(const runtime::NDArray::Container* lhs, const runtime::NDArray::Container* rhs,
                  SEqualReducer equal, bool compare_data);

/*!
 * \brief Hash NDArray.
 * \param arr The NDArray to compute the hash for.
 * \param hash_reduce A Reducer class to reduce the structural hash value.
 * See tvm/node/structural_hash.h.
 * \param hash_data Whether or not to hash ndarray raw data.
 */
void NDArrayHash(const runtime::NDArray::Container* arr, SHashReducer* hash_reduce, bool hash_data);

}  // namespace tvm

#endif  //  TVM_NODE_NDARRAY_HASH_EQUAL_H_
