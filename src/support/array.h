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
#include <tvm/ir/expr.h>
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

/*!
 * \brief Convert a tvm::runtime::Array to std::vector
 * \tparam TSrc The type of elements in the source Array
 * \tparam TDst The type of elements in the result vector
 * \return The result vector
 */
template <class TSrc, class TDst>
inline std::vector<TDst> AsVector(const Array<TSrc>& vec);

/*!
 * \brief Convert a std::vector to tvm::runtime::Array
 * \tparam TSrc The type of elements in the source vector
 * \tparam TDst The type of elements in the result Array
 * \return The result vector
 */
template <class TSrc, class TDst>
inline Array<TDst> AsArray(const std::vector<TSrc>& vec);

/*!
 * \brief Get the shape tuple as array
 * \param shape The shape tuple
 * \return An array of the shape tuple
 */
inline Array<Integer> AsArray(const ShapeTuple& shape) {
  Array<Integer> result;
  result.reserve(shape->size);
  for (ShapeTuple::index_type i : shape) {
    result.push_back(Integer(i));
  }
  return result;
}

/*!
 * \brief Concatenate a list of arrays into a single array
 * \tparam T The type of elements in the arrays
 * \tparam Iterator The type of the iterator into the list of arrays
 * \param begin The begin iterator to the array list
 * \param end The end iterator to the array list
 * \return The concatenated array
 */
template <class T, class Iterator>
inline Array<T> ConcatArrayList(Iterator begin, Iterator end) {
  int size = 0;
  for (Iterator it = begin; it != end; ++it) {
    size += (*it).size();
  }
  Array<T> result;
  result.reserve(size);
  for (Iterator it = begin; it != end; ++it) {
    const auto& item = *it;
    result.insert(result.end(), item.begin(), item.end());
  }
  return result;
}

/********** Implementation details of AsVector<TSrc, TDst> **********/

namespace details {

template <class TSrc, class TDst>
struct AsVectorImpl {};

template <class TSrc>
struct AsVectorImpl<TSrc, TSrc> {
  inline std::vector<TSrc> operator()(const Array<TSrc>& vec) const {
    return std::vector<TSrc>(vec.begin(), vec.end());
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int> {
  inline std::vector<int> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, int64_t> {
  inline std::vector<int64_t> operator()(const Array<TSrcObjectRef>& vec) const {
    std::vector<int64_t> results;
    for (const TSrcObjectRef& x : vec) {
      const auto* n = x.template as<IntImmNode>();
      ICHECK(n) << "TypeError: Expects IntImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};

template <class TSrcObjectRef>
struct AsVectorImpl<TSrcObjectRef, double> {
  inline std::vector<double> operator()(const Array<TSrcObjectRef>& array) const {
    std::vector<double> results;
    for (const TSrcObjectRef& x : array) {
      const auto* n = x.template as<FloatImmNode>();
      ICHECK(n) << "TypeError: Expects FloatImm, but gets: " << x->GetTypeKey();
      results.push_back(n->value);
    }
    return results;
  }
};
}  // namespace details

/********** Implementation details of AsArray<TSrc, TDst> **********/

namespace details {

template <class TSrc, class TDst>
struct AsArrayImpl {};

template <class TSrc>
struct AsArrayImpl<TSrc, TSrc> {
  inline Array<TSrc> operator()(const std::vector<TSrc>& vec) const {
    return Array<TSrc>(vec.begin(), vec.end());
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<int, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<int>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (int x : vec) {
      result.push_back(Integer(x));
    }
    return result;
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<int64_t, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<int64_t>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (int64_t x : vec) {
      result.push_back(Integer(x));
    }
    return result;
  }
};

template <class TDstObjectRef>
struct AsArrayImpl<double, TDstObjectRef> {
  inline Array<TDstObjectRef> operator()(const std::vector<double>& vec) const {
    Array<TDstObjectRef> result;
    result.reserve(vec.size());
    for (double x : vec) {
      result.push_back(FloatImm(tvm::DataType::Float(64), x));
    }
    return result;
  }
};

}  // namespace details

template <class TSrc, class TDst>
inline std::vector<TDst> AsVector(const Array<TSrc>& vec) {
  return details::AsVectorImpl<TSrc, TDst>()(vec);
}

template <class TSrc, class TDst>
inline Array<TDst> AsArray(const std::vector<TSrc>& vec) {
  return details::AsArrayImpl<TSrc, TDst>()(vec);
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_ARRAY_H_
