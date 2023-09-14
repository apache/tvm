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
 * \file src/contrib/msc/core/utils.h
 * \brief Common utilities for msc.
 */
#ifndef TVM_CONTRIB_MSC_CORE_UTILS_H_
#define TVM_CONTRIB_MSC_CORE_UTILS_H_

#include <tvm/ir/source_map.h>
#include <tvm/relax/expr.h>
#include <tvm/relay/expr.h>

#include <tuple>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {

using Expr = tvm::RelayExpr;
using RelaxCall = tvm::relax::Call;
using RelayCall = tvm::relay::Call;

class CommonUtils {
 public:
  /*!
   * \brief Check if the index is in range.
   * \return The valid index.
   */
  TVM_DLL static size_t GetIndex(int index, size_t max_size);

  /*!
   * \brief Check if the index is in range.
   * \return The valid indices.
   */
  TVM_DLL static std::vector<size_t> GetIndices(const std::vector<int>& indices, size_t max_size);
};

/*!
 * \brief Utils for String.
 */
class StringUtils {
 public:
  /*!
   * \brief Split the String into sub Strings.
   * \return The SubStrings.
   */
  TVM_DLL static const Array<String> Split(const String& src_string, const String& sep);

  /*!
   * \brief Join the SubStrings into String.
   * \return The String.
   */
  TVM_DLL static const String Join(const Array<String>& sub_strings, const String& joint);

  /*!
   * \brief Replace the substring old to new in String.
   * \return The replaced String.
   */
  TVM_DLL static const String Replace(const String& src_string, const String& old_str,
                                      const String& new_str);

  /*!
   * \brief Split the String into two sub Strings, only split by the frist seq.
   * \return The SubStrings.
   */
  TVM_DLL static const std::tuple<String, String> SplitOnce(const String& src_string,
                                                            const String& sep,
                                                            bool from_left = true);

  /*!
   * \brief Get the tokens between left and right.
   * \return The Tokens.
   */
  TVM_DLL static const Array<String> GetClosures(const String& src_string, const String& left,
                                                 const String& right);

  /*!
   * \brief Get the first token between left and right.
   * \return The Token.
   */
  TVM_DLL static const String GetClosureOnce(const String& src_string, const String& left,
                                             const String& right, bool from_left = true);

  /*!
   * \brief Change Object to String.
   * \return The String.
   */
  TVM_DLL static const String ToString(const runtime::ObjectRef& obj);

  /*!
   * \brief Compare String arrays.
   * \return Whether two array are same.
   */
  TVM_DLL static bool CompareArrays(const Array<String>& left, const Array<String>& right,
                                    int size = -1);
};

/*!
 * \brief Utils for Array.
 */
class ArrayUtils {
 public:
  /*!
   * \brief Replace the element old to new in Array.
   * \return The replaced Array.
   */
  template <typename T>
  TVM_DLL static const Array<T> Replace(const Array<T>& src_array, const T& old_ele,
                                        const T& new_ele) {
    Array<T> new_array;
    for (const auto& a : src_array) {
      if (a == old_ele) {
        new_array.push_back(new_ele);
      } else {
        new_array.push_back(a);
      }
    }
    return new_array;
  }

  /*!
   * \brief Find the index of element.
   * \return The index, -1 if not found.
   */
  template <typename T>
  TVM_DLL static int IndexOf(const std::vector<T>& array, const T& ele) {
    for (size_t i = 0; i < array.size(); i++) {
      if (array[i] == ele) {
        return i;
      }
    }
    return -1;
  }

  /*!
   * \brief Downcast elements in the array.
   * \return The downcasted array
   */
  template <typename T>
  TVM_DLL static const Array<T> Cast(const Array<PrimExpr>& src_array) {
    Array<T> new_array;
    for (const auto& s : src_array) {
      new_array.push_back(Downcast<T>(s));
    }
    return new_array;
  }
};

/*!
 * \brief Utils for Span.
 */
class SpanUtils {
 public:
  /*!
   * \brief Set <key>value</key> to the Span.
   * \return The new Span.
   */
  TVM_DLL static const Span SetAttr(const Span& span, const String& key, const String& value);

  /*!
   * \brief Get the value in <key>value</key> from the Span.
   * \return The value String.
   */
  TVM_DLL static const String GetAttr(const Span& span, const String& key);

  /*!
   * \brief Get all the key:value in format <key>value</key> from the Span.
   * \return The Attrs Map.
   */
  TVM_DLL static const Map<String, String> GetAttrs(const Span& span);
};

/*!
 * \brief Utils for Expr.
 */
class ExprUtils {
 public:
  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static const Array<String> GetInputTypes(const String& optype, size_t inputs_num,
                                                   bool as_relax);

  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static const Array<String> GetInputTypes(const RelaxCall& call);

  /*!
   * \brief Get the input types of call.
   * \return The input types.
   */
  TVM_DLL static const Array<String> GetInputTypes(const RelayCall& call);

  /*!
   * \brief Get the scalar value of ndarray.
   * \return The scalar value.
   */
  template <typename T>
  TVM_DLL static const T GetScalar(const runtime::NDArray& array, size_t i = 0) {
    if (array->dtype.code == kDLInt) {
      if (array->dtype.bits == 8) {
        return T(reinterpret_cast<int8_t*>(array->data)[i]);
      } else if (array->dtype.bits == 16) {
        return T(reinterpret_cast<int16_t*>(array->data)[i]);
      } else if (array->dtype.bits == 32) {
        return T(reinterpret_cast<int32_t*>(array->data)[i]);
      } else if (array->dtype.bits == 64) {
        return T(reinterpret_cast<int64_t*>(array->data)[i]);
      }
    } else if (array->dtype.code == kDLUInt) {
      if (array->dtype.bits == 1) {  // bool
        return T(reinterpret_cast<uint8_t*>(array->data)[i]);
      } else if (array->dtype.bits == 8) {
        return T(reinterpret_cast<uint8_t*>(array->data)[i]);
      } else if (array->dtype.bits == 16) {
        return T(reinterpret_cast<uint16_t*>(array->data)[i]);
      } else if (array->dtype.bits == 32) {
        return T(reinterpret_cast<uint32_t*>(array->data)[i]);
      } else if (array->dtype.bits == 64) {
        return T(reinterpret_cast<uint64_t*>(array->data)[i]);
      }
    } else if (array->dtype.code == kDLFloat) {
      if (array->dtype.bits == 32) {
        return T(reinterpret_cast<float*>(array->data)[i]);
      } else if (array->dtype.bits == 64) {
        return T(reinterpret_cast<double*>(array->data)[i]);
      }
    }
    LOG(FATAL) << "Failed to get scalar from array " << array;
  }

  /*!
   * \brief Get the scalar value of relax constant.
   * \return The scalar value.
   */
  template <typename T>
  TVM_DLL static const T GetScalar(const relax::Constant& constant, size_t i = 0) {
    return GetScalar<T>(constant->data, i);
  }

  /*!
   * \brief Get the scalar value of relay constant.
   * \return The scalar value.
   */
  template <typename T>
  TVM_DLL static const T GetScalar(const relay::Constant& constant, size_t i = 0) {
    return GetScalar<T>(constant->data, i);
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_UTILS_H_
