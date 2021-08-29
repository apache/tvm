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
#ifndef TVM_SUPPORT_ARRAY_H_
#define TVM_SUPPORT_ARRAY_H_
#include <tvm/runtime/container/array.h>

#include <vector>

namespace tvm {
namespace support {

/*!
 * \brief Checks if two arrays contain the same objects
 * \tparam T The type of objects in the array
 * \param a The first array
 * \param b The second array
 * \return A boolean indicating if they are the same
 */
template <class T>
inline bool ArrayWithSameContent(const Array<T>& a, const Array<T>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  int n = a.size();
  for (int i = 0; i < n; ++i) {
    if (!a[i].same_as(b[i])) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Checks if two arrays contain the same objects
 * \tparam T The type of objects in the array
 * \param a The first array
 * \param b The second array
 * \return A boolean indicating if they are the same
 */
template <class T>
inline bool ArrayWithSameContent(const std::vector<T*>& a, const std::vector<T*>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  int n = a.size();
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_ARRAY_H_
