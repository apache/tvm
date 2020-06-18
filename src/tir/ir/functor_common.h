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
 * \file tir/ir/functor_common.h
 * \brief Common utils for implementing functors
 */
#ifndef TVM_TIR_IR_FUNCTOR_COMMON_H_
#define TVM_TIR_IR_FUNCTOR_COMMON_H_

namespace tvm {
namespace tir {

// Implementation of Visitors
template<typename T, typename F>
inline void VisitArray(const Array<T>& arr, F fvisit) {
  for (size_t i = 0; i < arr.size(); i++) {
    fvisit(arr[i]);
  }
}

// Implementation of mutators
template<typename T, typename F>
inline Array<T> MutateArray(const Array<T>& arr,
                            F fmutate,
                            bool allow_copy_on_write = false) {
  if (allow_copy_on_write) {
    // if we allow copy on write, we can directly
    // call the inplace mutate function.
    const_cast<Array<T>&>(arr).MutateByApply(fmutate);
    return arr;
  } else {
    Array<T> copy = arr;
    copy.MutateByApply(fmutate);
    return copy;
  }
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_FUNCTOR_COMMON_H_
