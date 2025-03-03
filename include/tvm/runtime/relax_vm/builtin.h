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
 * \file tvm/runtime/relax_vm/builtin.h
 * \brief Builtin runtime APIs.
 */
#ifndef TVM_RUNTIME_RELAX_VM_BUILTIN_H_
#define TVM_RUNTIME_RELAX_VM_BUILTIN_H_

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief Op code used in built-in match-shape function.
 *
 * The function takes the following signature:

 * MatchShape(input_shape, shape_heap, n, c[0], r[0], c[1], r[1], ... c[n], r[n], err_ctx)
 *
 * This function provides runtime shape population and checking support for match-cast.
 * When a shape variable appears in the first time, we should load the shape and
 * populate the variable. When a shape variable already appears, we should
 * assert that it already equals an existing shape value.
 *
 * NOTE: It is OK to pass nullptr shape_heap if all code are AssertEqualToImm.
 */
enum class MatchShapeCode : int {
  /*!
   * \brief Perform an assertion that shape equals immediate.
   *
   * assert input_shape[i] == r[i]
   */
  kAssertEqualToImm = 0,
  /*!
   * \brief This is the first time we see a symbolic shape variable, store to heap.
   *
   * shape_heap[r[i]] = input_shape[i]
   */
  kStoreToHeap = 1,
  /*!
   * \brief skip and do not do anything.
   */
  kNoOp = 2,
  /*!
   * \brief Peform an assertion that the shape equals a loaded value.
   *
   * assert input_shape[i] == shape_heap[r[i]]
   */
  kAssertEqualToLoad = 3,
};

/*!
 * \brief Op code used in builtin function MakeShape.
 *
 * MakeShape(shape_heap, n, c[0], r[0], c[1], r[1], ... c[n], r[n]).
 *
 * \note It is OK to pass nullptr to shape_heap if all code are UseImm.
 */
enum class MakeShapeCode : int {
  /*! \brief Use the following r[i] as immediate shape value. */
  kUseImm = 0,
  /*!
   * \brief Load shape value from the shape_heap[[r[i]].
   */
  kLoadShape = 1,
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RELAX_VM_BUILTIN_H_
